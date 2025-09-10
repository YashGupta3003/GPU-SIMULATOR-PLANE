import sqlite3
import random
import time
import threading
from flask import Flask, render_template, request, jsonify

# ---- Config ----
DB_PATH = "telemetry.db"
NUM_GPUS = 4

# ---- Random but fixed throughput for this run ----
GPU_THROUGHPUTS = [random.randint(80, 150) for _ in range(NUM_GPUS)]

# Runtime state
GPU_ACTIVE = [0 for _ in range(NUM_GPUS)]
GPU_UTILS = [0.0 for _ in range(NUM_GPUS)]
GPU_BASE_TEMPS = [random.uniform(34.0, 42.0) for _ in range(NUM_GPUS)]
GPU_TEMPS = [bt for bt in GPU_BASE_TEMPS]
GPU_LOCKS = [threading.Lock() for _ in range(NUM_GPUS)]
GPU_BUSY_UNTIL = [0.0 for _ in range(NUM_GPUS)]
GPU_RUNNING_JOBS = [set() for _ in range(NUM_GPUS)]
GPU_QUEUED_JOBS = [list() for _ in range(NUM_GPUS)]

# Scheduler state
rr_index = 0
SCHEDULER_ORDER = ["random", "rr", "fcfs", "qlearn"]

# ---- DB init ----
def init_db():
    con = sqlite3.connect(DB_PATH, timeout=30)
    cur = con.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS jobs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            size INTEGER,
            gpu INTEGER,
            scheduler TEXT,
            status TEXT,
            duration REAL,
            submit_time REAL,
            batch_id INTEGER
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS q_table (
            state TEXT,
            action INTEGER,
            value REAL,
            PRIMARY KEY(state, action)
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS batch_groups (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            submit_time REAL
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS batches (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            group_id INTEGER,
            scheduler TEXT,
            submit_time REAL,
            finish_time REAL
        )
    """)

    cur.execute("PRAGMA table_info(batches)")
    cols = [r[1] for r in cur.fetchall()]
    if "group_id" not in cols:
        try:
            cur.execute("ALTER TABLE batches ADD COLUMN group_id INTEGER")
        except Exception:
            pass

    con.commit()
    con.close()

init_db()

# ---- Enhanced Q-Learning Functions ----
def get_gpu_state():
    """Get current state of all GPUs for Q-learning"""
    now = time.time()
    loads = []
    for i in range(NUM_GPUS):
        with GPU_LOCKS[i]:
            remaining_time = max(0.0, GPU_BUSY_UNTIL[i] - now)
            queue_size = len(GPU_QUEUED_JOBS[i])
            # Normalize load: combine remaining work time and queue length
            load = remaining_time + (queue_size * 2.0)  # rough estimate
            loads.append(load)
    return loads

def encode_state(job_size, gpu_loads):
    """Encode the current system state for Q-learning"""
    # Job size bucket (0-4)
    job_bucket = min(int(job_size // 500), 4)
    
    # GPU load buckets (0=light, 1=medium, 2=heavy for each GPU)
    max_load = max(gpu_loads) if gpu_loads else 1.0
    load_buckets = []
    for load in gpu_loads:
        if max_load < 0.1:
            bucket = 0
        else:
            normalized = load / max_load
            if normalized < 0.33:
                bucket = 0  # light
            elif normalized < 0.66:
                bucket = 1  # medium  
            else:
                bucket = 2  # heavy
        load_buckets.append(bucket)
    
    # Create state string: "job_bucket:load0,load1,load2,load3"
    state = f"{job_bucket}:{','.join(map(str, load_buckets))}"
    return state

def calculate_reward(job_size, gpu, wait_time, duration, gpu_loads_before, gpu_loads_after):
    """Calculate reward based on system efficiency improvement"""
    # Base penalty for job execution time
    time_penalty = -(wait_time + duration)
    
    # Load balancing reward: prefer actions that balance the system
    load_before = gpu_loads_before[gpu] if gpu < len(gpu_loads_before) else 0
    avg_load_before = sum(gpu_loads_before) / len(gpu_loads_before) if gpu_loads_before else 0
    
    # Reward for picking less loaded GPUs
    load_balance_reward = 0
    if avg_load_before > 0:
        relative_load = load_before / avg_load_before
        if relative_load < 0.8:  # GPU was underloaded
            load_balance_reward = 5.0
        elif relative_load > 1.2:  # GPU was overloaded
            load_balance_reward = -10.0
    
    # Efficiency bonus based on GPU throughput
    throughput_bonus = (GPU_THROUGHPUTS[gpu] / 100.0) * 2.0
    
    total_reward = time_penalty + load_balance_reward + throughput_bonus
    return total_reward

# ---- Scheduling ----
def pick_gpu(job_size, scheduler):
    global rr_index
    if scheduler == "random":
        return random.randint(0, NUM_GPUS - 1)
    elif scheduler == "rr":
        g = rr_index
        rr_index = (rr_index + 1) % NUM_GPUS
        return g
    elif scheduler == "fcfs":
        mins = min(GPU_ACTIVE)
        candidates = [i for i, a in enumerate(GPU_ACTIVE) if a == mins]
        return random.choice(candidates)
    elif scheduler == "qlearn":
        gpu_loads = get_gpu_state()
        state = encode_state(job_size, gpu_loads)
        
        con = sqlite3.connect(DB_PATH)
        cur = con.cursor()
        cur.execute("SELECT action, value FROM q_table WHERE state=?", (state,))
        rows = cur.fetchall()
        con.close()
        
        if rows and random.random() > 0.1:  # 90% exploitation, 10% exploration
            best_val = max(rows, key=lambda r: r[1])[1]
            candidates = [r[0] for r in rows if r[1] >= best_val - 0.1]  # Allow some tolerance
            return random.choice(candidates)
        else:  # Exploration: pick based on load balancing
            # Find GPUs with lowest current load
            min_load = min(gpu_loads)
            light_gpus = [i for i, load in enumerate(gpu_loads) if load <= min_load + 1.0]
            return random.choice(light_gpus)
    return 0

def update_q_table(job_size, gpu, wait_time, duration, gpu_loads_before, gpu_loads_after):
    """Update Q-table with improved reward function"""
    state = encode_state(job_size, gpu_loads_before)
    reward = calculate_reward(job_size, gpu, wait_time, duration, gpu_loads_before, gpu_loads_after)
    
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("SELECT value FROM q_table WHERE state=? AND action=?", (state, gpu))
    row = cur.fetchone()
    
    learning_rate = 0.1
    if row is None:
        cur.execute("INSERT INTO q_table(state, action, value) VALUES(?,?,?)", (state, gpu, reward))
    else:
        old_value = row[0]
        new_value = old_value + learning_rate * (reward - old_value)
        cur.execute("UPDATE q_table SET value=? WHERE state=? AND action=?", (new_value, state, gpu))
    
    con.commit()
    con.close()

# ---- Job execution ----
def run_job(job_id, job_size, gpu, scheduler, batch_id):
    throughput = max(1.0, GPU_THROUGHPUTS[gpu])
    duration = float(job_size) / throughput

    # Capture state before job execution for Q-learning
    gpu_loads_before = get_gpu_state() if scheduler == "qlearn" else None
    
    now = time.time()
    with GPU_LOCKS[gpu]:
        start_time = max(now, GPU_BUSY_UNTIL[gpu])
        wait_time = start_time - now
        GPU_BUSY_UNTIL[gpu] = start_time + duration
        GPU_QUEUED_JOBS[gpu].append(job_id)

    if wait_time > 0:
        time.sleep(wait_time)

    with GPU_LOCKS[gpu]:
        if job_id in GPU_QUEUED_JOBS[gpu]:
            GPU_QUEUED_JOBS[gpu].remove(job_id)
        GPU_RUNNING_JOBS[gpu].add(job_id)
        GPU_ACTIVE[gpu] += 1
        GPU_TEMPS[gpu] = min(95.0, GPU_TEMPS[gpu] + random.uniform(2.0, 6.0))

    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("UPDATE jobs SET status=? WHERE id=?", ("running", job_id))
    con.commit()
    con.close()

    time.sleep(duration)

    with GPU_LOCKS[gpu]:
        GPU_ACTIVE[gpu] = max(0, GPU_ACTIVE[gpu] - 1)
        if job_id in GPU_RUNNING_JOBS[gpu]:
            GPU_RUNNING_JOBS[gpu].remove(job_id)
        GPU_TEMPS[gpu] = max(GPU_BASE_TEMPS[gpu], GPU_TEMPS[gpu] - random.uniform(1.0, 3.0))

    finish = time.time()
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("UPDATE jobs SET status=?, duration=? WHERE id=?", ("finished", duration, job_id))
    cur.execute("SELECT COUNT(*) FROM jobs WHERE batch_id=? AND status!='finished'", (batch_id,))
    remaining = cur.fetchone()[0]
    if remaining == 0:
        cur.execute("UPDATE batches SET finish_time=? WHERE id=?", (finish, batch_id))
    con.commit()
    con.close()

    # Update Q-table after job completion
    if scheduler == "qlearn" and gpu_loads_before is not None:
        gpu_loads_after = get_gpu_state()
        update_q_table(job_size, gpu, wait_time, duration, gpu_loads_before, gpu_loads_after)

# ---- Batch runner ----
def run_batch_for_schedulers(group_id, sizes, schedulers):
    for sched in schedulers:
        now = time.time()
        con = sqlite3.connect(DB_PATH)
        cur = con.cursor()
        cur.execute("INSERT INTO batches (group_id, scheduler, submit_time, finish_time) VALUES (?, ?, ?, NULL)",
                    (group_id, sched, now))
        batch_id = cur.lastrowid
        con.commit()
        con.close()

        job_threads = []
        for s in sizes:
            gpu = pick_gpu(s, scheduler=sched)
            submit_time = time.time()
            con = sqlite3.connect(DB_PATH)
            cur = con.cursor()
            cur.execute(
                "INSERT INTO jobs (size,gpu,scheduler,status,duration,submit_time,batch_id) VALUES (?,?,?,?,?,?,?)",
                (s, gpu, sched, "queued", None, submit_time, batch_id))
            job_id = cur.lastrowid
            con.commit()
            con.close()

            t = threading.Thread(target=run_job, args=(job_id, s, gpu, sched, batch_id), daemon=True)
            t.start()
            job_threads.append(t)

        for t in job_threads:
            t.join()

# ---- Flask app ----
app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html", num_gpus=NUM_GPUS, throughputs=GPU_THROUGHPUTS)

@app.route("/run_batch", methods=["POST"])
def run_batch():
    data = request.get_json() or {}
    sizes = data.get("sizes", [])
    if not sizes:
        return jsonify({"error": "No sizes"}), 400

    schedulers = SCHEDULER_ORDER
    now = time.time()
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("INSERT INTO batch_groups (submit_time) VALUES (?)", (now,))
    group_id = cur.lastrowid
    con.commit()
    con.close()

    t = threading.Thread(target=run_batch_for_schedulers, args=(group_id, sizes, schedulers), daemon=True)
    t.start()

    return jsonify({"ok": True, "group_id": group_id})

@app.route("/jobs")
def jobs():
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("""
        SELECT id, size, gpu, scheduler, status, duration, submit_time, batch_id
        FROM jobs
        ORDER BY id DESC
        LIMIT 200
    """)
    rows = cur.fetchall()
    con.close()
    jobs = []
    for r in rows:
        jobs.append({
            "id": r[0], "size": r[1], "gpu_index": r[2], "scheduler": r[3],
            "status": r[4], "duration": r[5], "submit_time": r[6], "batch_id": r[7]
        })
    return jsonify(jobs)

@app.route("/metrics/latest")
def metrics_latest():
    now = time.time()
    with threading.Lock():
        for i in range(NUM_GPUS):
            with GPU_LOCKS[i]:
                remaining = max(0.0, GPU_BUSY_UNTIL[i] - now)
                util3 = 100.0 if remaining >= 3.0 else (remaining / 3.0) * 100.0
                GPU_UTILS[i] = float(max(0.0, min(100.0, util3)))
                if remaining > 0.0:
                    GPU_TEMPS[i] = min(95.0, GPU_TEMPS[i] + 0.05 * (GPU_UTILS[i] / 100.0) * 5.0 + random.uniform(0.0, 0.4))
                else:
                    GPU_TEMPS[i] = max(GPU_BASE_TEMPS[i], GPU_TEMPS[i] - 0.6)

    data = []
    for i in range(NUM_GPUS):
        with GPU_LOCKS[i]:
            util = float(GPU_UTILS[i])
            temp = float(GPU_TEMPS[i])
        data.append({"gpu_index": i, "utilization": util, "temperature_c": temp})
    return jsonify({"data": data})

@app.route("/busy")
def busy_status():
    now = time.time()
    out = []
    for i in range(NUM_GPUS):
        with GPU_LOCKS[i]:
            remaining = max(0.0, GPU_BUSY_UNTIL[i] - now)
            util3 = 100.0 if remaining >= 3.0 else (remaining / 3.0) * 100.0
            queued = list(GPU_QUEUED_JOBS[i])
            running = sorted([int(j) for j in list(GPU_RUNNING_JOBS[i])])
            out.append({
                "gpu_index": i, "busy": remaining > 0.0, "remaining_sec": remaining,
                "window_util_percent": max(0.0, min(100.0, util3)),
                "throughput": GPU_THROUGHPUTS[i],
                "running_jobs": running, "queued_jobs": queued
            })
    return jsonify({"data": out})

@app.route("/batch_performance")
def batch_performance():
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("""
        SELECT id, group_id, scheduler, submit_time, finish_time,
               (finish_time - submit_time) as batch_time
        FROM batches
        WHERE finish_time IS NOT NULL
        ORDER BY group_id DESC, id DESC
        LIMIT 200
    """)
    rows = cur.fetchall()
    con.close()
    return jsonify({"data": [
        {"id": r[0], "group_id": r[1], "scheduler": r[2],
         "submit_time": r[3], "finish_time": r[4], "batch_time": r[5]}
        for r in rows if r[3] and r[4]
    ]})

@app.route("/qtable")
def qtable():
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("SELECT state, action, value FROM q_table ORDER BY value DESC")
    rows = cur.fetchall()
    con.close()
    return jsonify({"data": [{"state": r[0], "action": r[1], "value": r[2]} for r in rows]})

@app.route("/throughputs")
def get_throughputs():
    return jsonify({"data": GPU_THROUGHPUTS})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)