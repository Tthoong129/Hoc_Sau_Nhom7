const socket = io();
const $ = id => document.getElementById(id);
let videoFps = 30;
let totalFrames = 0;
let currentFolder = "session";
let selectedVideoFile = null;
setInterval(() => {
    const now = new Date();
    $("clock").textContent = now.toLocaleDateString("vi") + "  " +
        now.toLocaleTimeString("vi", { hour12: false });
}, 1000);
let uploadedVideoPath = null;

$("source-select").addEventListener("change", e => {
    const isVideo = e.target.value === "video";
    $("btn-browse-video").style.display = isVideo ? "inline-block" : "none";
    $("upload-status").style.display = isVideo ? "inline-block" : "none";
    $("seek-group").style.display = isVideo ? "flex" : "none";
    if (isVideo && uploadedVideoPath) {
        triggerVideoStart(uploadedVideoPath);
    }
});

$("btn-browse-video").addEventListener("click", () => {
    $("video-upload-input").click();
});

$("video-upload-input").addEventListener("change", async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    
    $("upload-status").textContent = "Đang tải lên...";
    const formData = new FormData();
    formData.append("file", file);
    
    try {
        const res = await fetch("/api/upload_video", {
            method: "POST",
            body: formData
        });
        const data = await res.json();
        if (data.path) {
            uploadedVideoPath = data.path;
            $("upload-status").textContent = file.name;
            triggerVideoStart(uploadedVideoPath);
        } else {
            $("upload-status").textContent = "Lỗi tải lên";
        }
    } catch (err) {
        $("upload-status").textContent = "Lỗi kết nối";
    }
});

function triggerVideoStart(path) {
    if (path) {
        socket.emit("start", {
            source: "video",
            video_path: path
        });
    }
}

$("btn-start").addEventListener("click", () => {
    const source = $("source-select").value;
    if (source === "video") {
        if (!uploadedVideoPath) {
            alert("Vui lòng chọn file video trước!");
            return;
        }
        triggerVideoStart(uploadedVideoPath);
    } else {
        socket.emit("start", { source: "camera" });
    }
});
$("btn-stop").addEventListener("click", () => socket.emit("stop"));
$("btn-pause").addEventListener("click", () => socket.emit("pause"));
$("btn-shot").addEventListener("click", () => socket.emit("manual_shot"));
$("btn-debug").addEventListener("click", () => socket.emit("toggle_debug"));
$("btn-export").addEventListener("click", async () => {
    const res = await fetch("/api/export");
    const data = await res.json();
    if (data.ok) addLog("Báo cáo đã lưu: " + data.path, "INFO");
});
$("btn-clear-log").addEventListener("click", () => {
    $("alert-log").innerHTML = "";
});
$("conf-slider").addEventListener("input", e => {
    const val = (e.target.value / 100).toFixed(2);
    $("conf-value").textContent = val;
    socket.emit("set_conf", { value: parseFloat(val) });
});
$("speed-select").addEventListener("change", e => {
    socket.emit("set_speed", { value: parseFloat(e.target.value) });
});
$("seek-slider").addEventListener("input", e => {
    socket.emit("seek", { pos: parseInt(e.target.value) });
});
$("btn-auto-capture").addEventListener("click", () => {
    socket.emit("toggle_capture");
});
document.querySelectorAll(".tab-btn").forEach(btn => {
    btn.addEventListener("click", () => {
        document.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
        document.querySelectorAll(".tab-content").forEach(c => c.classList.remove("active"));
        btn.classList.add("active");
        $("tab-" + btn.dataset.tab).classList.add("active");
        if (btn.dataset.tab === "gallery") loadGallery(currentFolder);
    });
});
document.querySelectorAll(".gallery-tab").forEach(btn => {
    btn.addEventListener("click", () => {
        document.querySelectorAll(".gallery-tab").forEach(b => b.classList.remove("active"));
        btn.classList.add("active");
        currentFolder = btn.dataset.folder;
        loadGallery(currentFolder);
    });
});
$("lightbox").addEventListener("click", () => $("lightbox").style.display = "none");
$("lightbox-close").addEventListener("click", () => $("lightbox").style.display = "none");
function openLightbox(src) {
    $("lightbox-img").src = src;
    $("lightbox").style.display = "flex";
}
async function loadGallery(folder) {
    const res = await fetch(`/api/screenshots/${folder}`);
    const files = await res.json();
    const grid = $("gallery-grid");
    grid.innerHTML = "";
    if (files.length === 0) {
        grid.innerHTML = '<p style="color:var(--muted);text-align:center;grid-column:1/-1;padding:30px;font-size:13px;">Chưa có ảnh</p>';
        return;
    }
    files.forEach(f => {
        const div = document.createElement("div");
        div.className = "gallery-item";
        const src = `/screenshots/${folder}/${f}`;
        div.innerHTML = `<img src="${src}" loading="lazy"><div class="gallery-item-name">${f}</div>`;
        div.addEventListener("click", () => openLightbox(src));
        grid.appendChild(div);
    });
}
socket.on("frame", data => {
    const feed = $("video-feed");
    const ph = $("video-placeholder");
    feed.src = "data:image/jpeg;base64," + data.data;
    feed.style.display = "block";
    ph.style.display = "none";
});
socket.on("started", data => {
    setRunning(true);
    addLog("Bắt đầu giám sát (" + data.source + ")", "INFO");
});
socket.on("stopped", () => {
    setRunning(false);
    addLog("Đã dừng giám sát", "INFO");
});
socket.on("video_ended", () => {
    setRunning(false);
    addLog("Video kết thúc", "INFO");
});
socket.on("paused", data => {
    const badge = $("status-badge");
    const btnP = $("btn-pause");
    if (data.paused) {
        badge.className = "badge paused";
        badge.textContent = "TẠM DỪNG";
        btnP.textContent = "▶ TIẾP TỤC";
    } else {
        badge.className = "badge running";
        badge.textContent = "ĐANG GIÁM SÁT";
        btnP.textContent = "⏸ TẠM DỪNG";
    }
});
socket.on("fps", data => {
    $("stat-fps").textContent = data.value;
});
socket.on("alert", data => {
    addLog(data.msg + (data.source ? ` [${data.source}]` : ""), data.severity, data.screenshot);
    updateStats();
});
socket.on("video_info", data => {
    totalFrames = data.total_frames;
    videoFps = data.fps;
    $("seek-slider").max = totalFrames;
    $("seek-group").style.display = "flex";
    updateTimeDisplay(0);
});
socket.on("video_pos", data => {
    $("seek-slider").value = data.pos;
    updateTimeDisplay(data.pos);
});
socket.on("capture_status", data => {
    const btn = $("btn-auto-capture");
    btn.classList.toggle("active", data.auto_capture);
    btn.textContent = data.auto_capture ? "ON" : "OFF";
});
socket.on("debug_status", data => {
    const btn = $("btn-debug");
    btn.style.background = data.debug ? "var(--warn)" : "";
    btn.style.color = data.debug ? "white" : "";
});
socket.on("screenshot_taken", data => {
    addLog("📸 Đã chụp: " + data.filename, "INFO");
    if (currentFolder === "session") loadGallery("session");
});
socket.on("error", data => {
    addLog("❌ " + data.msg, "HIGH");
});
function setRunning(isRunning) {
    const badge = $("status-badge");
    $("btn-start").disabled = isRunning;
    $("btn-stop").disabled = !isRunning;
    $("btn-pause").disabled = !isRunning;
    if (isRunning) {
        badge.className = "badge running";
        badge.textContent = "ĐANG GIÁM SÁT";
        $("btn-pause").textContent = "⏸ TẠM DỪNG";
    } else {
        badge.className = "badge offline";
        badge.textContent = "OFFLINE";
        $("video-feed").style.display = "none";
        $("video-placeholder").style.display = "block";
    }
}
function addLog(msg, severity = "INFO", screenshot = null) {
    const log = $("alert-log");
    const now = new Date().toLocaleTimeString("vi", { hour12: false });
    const div = document.createElement("div");
    div.className = "log-entry " + severity;
    let html = `<span class="log-time">[${now}]</span> ${msg}`;
    if (screenshot) {
        html += ` <span class="log-screenshot" onclick="openLightbox('/screenshots/session/${screenshot}')">📷 xem</span>`;
    }
    div.innerHTML = html;
    log.appendChild(div);
    log.scrollTop = log.scrollHeight;
    const totalEl = $("stat-total");
    if (severity === "HIGH" || severity === "MEDIUM") {
        totalEl.textContent = parseInt(totalEl.textContent) + 1;
    }
    if (severity === "HIGH") {
        const el = $("stat-high");
        el.textContent = parseInt(el.textContent) + 1;
    }
    if (severity === "MEDIUM") {
        const el = $("stat-medium");
        el.textContent = parseInt(el.textContent) + 1;
    }
}
async function updateStats() {
    const res = await fetch("/api/stats");
    const data = await res.json();
    $("stat-total").textContent = data.total;
    $("stat-high").textContent = data.high;
    $("stat-medium").textContent = data.medium;
}
function updateTimeDisplay(pos) {
    const cur = Math.floor(pos / videoFps);
    const tot = Math.floor(totalFrames / videoFps);
    const fmt = s => {
        const m = Math.floor(s / 60);
        const sec = s % 60;
        return `${String(m).padStart(2, "0")}:${String(sec).padStart(2, "0")}`;
    };
    $("time-display").textContent = fmt(cur) + " / " + fmt(tot);
}
