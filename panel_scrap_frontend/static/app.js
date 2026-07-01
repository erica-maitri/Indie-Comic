/* -------------------------------------------------------------
   PANEL SCRAP - FRONTEND LOGIC & INTERACTIVE PROTOTYPE
   ------------------------------------------------------------- */

document.addEventListener("DOMContentLoaded", () => {
    
    // ---------------------------------------------------------
    // DOM ELEMENTS
    // ---------------------------------------------------------
    
    // Navigation Tabs
    const tabWriter = document.getElementById("tab-writer");
    const tabStoryboard = document.getElementById("tab-storyboard");
    const tabExport = document.getElementById("tab-export");
    
    // View Sections
    const writerView = document.getElementById("writer-view");
    const storyboardView = document.getElementById("storyboard-view");
    const exportView = document.getElementById("export-view");
    
    // Writer's Room Controls
    const premiseInput = document.getElementById("premise-input");
    const charCount = document.getElementById("char-count");
    const panelSlider = document.getElementById("panel-slider");
    const panelCountVal = document.getElementById("panel-count-val");
    const modeQuick = document.getElementById("mode-quick");
    const modeGuided = document.getElementById("mode-guided");
    const styleCards = document.querySelectorAll(".style-card");
    const styleCustomBtn = document.getElementById("style-custom-btn");
    const generateOutlineBtn = document.getElementById("generate-outline-btn");
    const generationLoader = document.getElementById("generation-loader");
    const loaderStatusText = document.getElementById("loader-status-text");
    const loaderProgress = document.getElementById("loader-progress");
    const loaderTimeLeft = document.getElementById("loader-time-left");
    
    // Custom Style Modal
    const styleModal = document.getElementById("style-modal");
    const closeModalBtn = document.getElementById("close-modal-btn");
    const saveStyleBtn = document.getElementById("save-style-btn");
    const uploadZone = document.getElementById("upload-zone");
    const modalFileInput = document.getElementById("modal-file-input");
    const customStylePrompt = document.getElementById("custom-style-prompt");
    
    // Storyboard Controls
    const drawPanelsBtn = document.getElementById("draw-panels-btn");
    const regenerateOutlineBtn = document.getElementById("regenerate-outline-btn");
    const consoleLogsContainer = document.getElementById("console-logs-container");
    const emotionCanvas = document.getElementById("emotion-canvas");
    
    // Export & Publish Controls
    const exportActionBtns = document.querySelectorAll(".export-action-btn");
    const publishProjectBtn = document.getElementById("publish-project-btn");
    const successModal = document.getElementById("success-modal");
    const successCloseBtn = document.getElementById("success-close-btn");

    // ---------------------------------------------------------
    // TAB NAVIGATION
    // ---------------------------------------------------------
    function switchTab(viewId) {
        // Remove active class from all tabs & sections
        [tabWriter, tabStoryboard, tabExport].forEach(tab => tab.classList.remove("active"));
        [writerView, storyboardView, exportView].forEach(view => view.classList.remove("active"));
        
        if (viewId === "writer") {
            tabWriter.classList.add("active");
            writerView.classList.add("active");
        } else if (viewId === "storyboard") {
            tabStoryboard.classList.add("active");
            storyboardView.classList.add("active");
            // Draw/redraw the emotion arc when tab becomes visible
            setTimeout(initEmotionArc, 50);
        } else if (viewId === "export") {
            tabExport.classList.add("active");
            exportView.classList.add("active");
        }
        
        // Scroll back to top of main scroll area
        document.querySelector(".app-content").scrollTop = 0;
    }
    
    tabWriter.addEventListener("click", () => switchTab("writer"));
    tabStoryboard.addEventListener("click", () => switchTab("storyboard"));
    tabExport.addEventListener("click", () => switchTab("export"));

    // ---------------------------------------------------------
    // STORY PREMISE CHARACTER COUNTER
    // ---------------------------------------------------------
    premiseInput.addEventListener("input", (e) => {
        const textLength = e.target.value.length;
        charCount.textContent = textLength;
        if (textLength >= 450) {
            charCount.style.color = "var(--color-accent-red)";
            charCount.style.fontWeight = "700";
        } else {
            charCount.style.color = "var(--color-text-muted)";
            charCount.style.fontWeight = "500";
        }
    });

    // ---------------------------------------------------------
    // PANEL COUNT SLIDER
    // ---------------------------------------------------------
    panelSlider.addEventListener("input", (e) => {
        panelCountVal.textContent = e.target.value;
    });

    // ---------------------------------------------------------
    // CREATION MODE TOGGLE
    // ---------------------------------------------------------
    [modeQuick, modeGuided].forEach(btn => {
        btn.addEventListener("click", () => {
            modeQuick.classList.remove("active");
            modeGuided.classList.remove("active");
            btn.classList.add("active");
            
            addConsoleLog(`[SYSTEM] Switched creation mode to: ${btn.textContent.trim()}`);
        });
    });

    // ---------------------------------------------------------
    // ART STYLE SELECTION & MODAL
    // ---------------------------------------------------------
    styleCards.forEach(card => {
        card.addEventListener("click", () => {
            if (card.id === "style-custom-btn") {
                // Open Custom Style modal
                styleModal.style.display = "flex";
                return;
            }
            
            styleCards.forEach(c => c.classList.remove("selected"));
            card.classList.add("selected");
            
            const selectedStyleName = card.querySelector(".style-badge").textContent;
            addConsoleLog(`[SYSTEM] Style changed to: ${selectedStyleName}`);
        });
    });

    // Modal Close
    closeModalBtn.addEventListener("click", () => {
        styleModal.style.display = "none";
    });

    // Save Custom Style
    saveStyleBtn.addEventListener("click", () => {
        const promptText = customStylePrompt.value.trim();
        if (promptText) {
            // Update custom card badge to show prompt excerpt
            const customBadge = styleCustomBtn.querySelector(".style-badge");
            customBadge.textContent = promptText.substring(0, 10).toUpperCase() + "...";
            
            // Mark custom card as active
            styleCards.forEach(c => c.classList.remove("selected"));
            styleCustomBtn.classList.add("selected");
            
            addConsoleLog(`[SYSTEM] Custom style set: "${promptText}"`);
        }
        styleModal.style.display = "none";
    });

    // Reference File Upload Drag/Drop Mock
    uploadZone.addEventListener("click", () => {
        modalFileInput.click();
    });

    modalFileInput.addEventListener("change", (e) => {
        if (e.target.files && e.target.files[0]) {
            const filename = e.target.files[0].name;
            uploadZone.querySelector("span").textContent = `Selected: ${filename}`;
            addConsoleLog(`[SYSTEM] Referenced image loaded: ${filename}`);
        }
    });

    // ---------------------------------------------------------
    // STORY OUTLINE GENERATION SIMULATION
    // ---------------------------------------------------------
    let generationInterval = null;
    generateOutlineBtn.addEventListener("click", () => {
        // Disable outline button to prevent multiple triggers
        generateOutlineBtn.parentElement.style.display = "none";
        generationLoader.style.display = "block";
        
        let timeLeft = 15;
        let progress = 0;
        loaderTimeLeft.textContent = timeLeft;
        loaderProgress.style.width = "0%";
        
        const statusSteps = [
            { threshold: 0, text: "Analyzing story premise..." },
            { threshold: 25, text: "Mapping character emotional beats..." },
            { threshold: 50, text: "Designing scene intensity graphs..." },
            { threshold: 75, text: "Drafting layout panels..." },
            { threshold: 90, text: "Assembling storyboard elements..." }
        ];

        generationInterval = setInterval(() => {
            progress += 100 / 150; // Progress matches 15 seconds subdivided by 100ms ticks
            loaderProgress.style.width = `${Math.min(progress, 100)}%`;
            
            // Time remaining updates every 1s
            if (Math.round(progress) % 7 === 0) {
                const calculatedTime = Math.max(15 - Math.round((progress / 100) * 15), 0);
                loaderTimeLeft.textContent = calculatedTime;
            }

            // Status message updating based on thresholds
            const currentStatus = statusSteps.reduce((acc, step) => {
                if (progress >= step.threshold) return step.text;
                return acc;
            }, statusSteps[0].text);
            loaderStatusText.textContent = currentStatus;

            if (progress >= 100) {
                clearInterval(generationInterval);
                
                // Hide loader and show button back
                generationLoader.style.display = "none";
                generateOutlineBtn.parentElement.style.display = "block";
                
                // Route to Storyboard
                switchTab("storyboard");
                addConsoleLog(`[STORY] Successfully generated 6-panel outline matching Noir style!`);
            }
        }, 100);
    });

    // ---------------------------------------------------------
    // STORYBOARD: INTERACTIVE EMOTION ARC GRAPH (HTML5 CANVAS)
    // ---------------------------------------------------------
    let graphPoints = [
        { x: 30, y: 100, label: "Start", draggable: false },
        { x: 120, y: 70, label: "Build-up", draggable: true },
        { x: 180, y: 40, label: "Climax", draggable: true },
        { x: 230, y: 80, label: "Fall", draggable: true },
        { x: 280, y: 20, label: "Ending", draggable: false }
    ];
    let selectedPointIndex = -1;

    function initEmotionArc() {
        const ctx = emotionCanvas.getContext("2d");
        
        // Draw the current state
        drawChart();
        
        // Handle Canvas mouse drag interactions
        emotionCanvas.onmousedown = (e) => {
            const rect = emotionCanvas.getBoundingClientRect();
            // Calculate scale in case of styling scaling
            const scaleX = emotionCanvas.width / rect.width;
            const scaleY = emotionCanvas.height / rect.height;
            const mouseX = (e.clientX - rect.left) * scaleX;
            const mouseY = (e.clientY - rect.top) * scaleY;
            
            // Check if clicked close to a draggable point
            selectedPointIndex = graphPoints.findIndex(pt => {
                if (!pt.draggable) return false;
                const dist = Math.sqrt(Math.pow(pt.x - mouseX, 2) + Math.pow(pt.y - mouseY, 2));
                return dist < 12; // Click radius
            });
            
            if (selectedPointIndex !== -1) {
                addConsoleLog(`[UI] Editing emotion intensity at: ${graphPoints[selectedPointIndex].label}`);
            }
        };

        emotionCanvas.onmousemove = (e) => {
            if (selectedPointIndex === -1) return;
            const rect = emotionCanvas.getBoundingClientRect();
            const scaleY = emotionCanvas.height / rect.height;
            const mouseY = (e.clientY - rect.top) * scaleY;
            
            // Clamp Y inside canvas boundaries (keeping some margins)
            graphPoints[selectedPointIndex].y = Math.max(15, Math.min(mouseY, 115));
            drawChart();
        };

        emotionCanvas.onmouseup = () => {
            if (selectedPointIndex !== -1) {
                const intensityVal = Math.round(((130 - graphPoints[selectedPointIndex].y) / 130) * 100);
                addConsoleLog(`[SYSTEM] Graph point "${graphPoints[selectedPointIndex].label}" intensity set to ${intensityVal}%`);
                selectedPointIndex = -1;
            }
        };
        
        // Touch events for mobile support
        emotionCanvas.ontouchstart = (e) => {
            if (e.touches.length > 0) {
                const touch = e.touches[0];
                const rect = emotionCanvas.getBoundingClientRect();
                const scaleX = emotionCanvas.width / rect.width;
                const scaleY = emotionCanvas.height / rect.height;
                const mouseX = (touch.clientX - rect.left) * scaleX;
                const mouseY = (touch.clientY - rect.top) * scaleY;
                
                selectedPointIndex = graphPoints.findIndex(pt => {
                    if (!pt.draggable) return false;
                    const dist = Math.sqrt(Math.pow(pt.x - mouseX, 2) + Math.pow(pt.y - mouseY, 2));
                    return dist < 15;
                });
            }
        };
        
        emotionCanvas.ontouchmove = (e) => {
            if (selectedPointIndex === -1 || e.touches.length === 0) return;
            const touch = e.touches[0];
            const rect = emotionCanvas.getBoundingClientRect();
            const scaleY = emotionCanvas.height / rect.height;
            const mouseY = (touch.clientY - rect.top) * scaleY;
            
            graphPoints[selectedPointIndex].y = Math.max(15, Math.min(mouseY, 115));
            drawChart();
            e.preventDefault();
        };

        emotionCanvas.ontouchend = () => {
            selectedPointIndex = -1;
        };

        function drawChart() {
            ctx.clearRect(0, 0, emotionCanvas.width, emotionCanvas.height);
            
            // Draw axis markers (Background grid lines)
            ctx.strokeStyle = "#EBE6DC";
            ctx.lineWidth = 1;
            for (let y = 30; y < emotionCanvas.height; y += 30) {
                ctx.beginPath();
                ctx.moveTo(0, y);
                ctx.lineTo(emotionCanvas.width, y);
                ctx.stroke();
            }

            // Draw Spline Curve (Emotion Line)
            ctx.beginPath();
            ctx.moveTo(graphPoints[0].x, graphPoints[0].y);
            
            // Quadratic/Bézier approximation of path spline
            for (let i = 0; i < graphPoints.length - 1; i++) {
                const xc = (graphPoints[i].x + graphPoints[i+1].x) / 2;
                const yc = (graphPoints[i].y + graphPoints[i+1].y) / 2;
                ctx.quadraticCurveTo(graphPoints[i].x, graphPoints[i].y, xc, yc);
            }
            ctx.lineTo(graphPoints[graphPoints.length - 1].x, graphPoints[graphPoints.length - 1].y);
            
            ctx.strokeStyle = "var(--color-accent-red)";
            ctx.lineWidth = 4;
            ctx.stroke();
            
            // Draw each control point dot
            graphPoints.forEach(pt => {
                ctx.beginPath();
                ctx.arc(pt.x, pt.y, pt.draggable ? 6 : 4, 0, 2 * Math.PI);
                ctx.fillStyle = pt.label === "Climax" ? "#7A2012" : "var(--color-border)";
                ctx.fill();
                
                // Add a border if it's draggable
                if (pt.draggable) {
                    ctx.strokeStyle = "var(--color-accent-yellow)";
                    ctx.lineWidth = 2.5;
                    ctx.stroke();
                }
                
                // Label above Climax
                if (pt.label === "Climax") {
                    ctx.font = "bold 10px 'Space Grotesk', sans-serif";
                    ctx.fillStyle = "var(--color-text-dark)";
                    ctx.fillText("Climax", pt.x - 18, pt.y - 12);
                }
            });
        }
    }

    // ---------------------------------------------------------
    // STORYBOARD SIMULATION: DRAW COMIC PANELS
    // ---------------------------------------------------------
    drawPanelsBtn.addEventListener("click", () => {
        addConsoleLog(`[SD-BACKEND] Initiating sketch generation sequence...`);
        
        // Scroll to terminal to see action
        document.querySelector(".system-log-console").scrollIntoView({ behavior: 'smooth' });
        
        // Progressively trigger sketches loading
        simulatePanelDrawProgress(1, 1000, () => {
            simulatePanelDrawProgress(2, 2800, () => {
                addConsoleLog(`[SYSTEM] Sketch render compilation success! Navigating to Export page...`);
                // Auto switch to export view
                setTimeout(() => {
                    switchTab("export");
                }, 1000);
            });
        });
    });

    function simulatePanelDrawProgress(panelId, initialDelay, callback) {
        setTimeout(() => {
            addConsoleLog(`[SD-MODEL] Loading LORA weights for Style: Noir...`);
            
            setTimeout(() => {
                addConsoleLog(`[SD-MODEL] Rendering scene panel ${panelId} matching descriptor...`);
                
                setTimeout(() => {
                    const frame = document.getElementById(`panel-img-frame-${panelId}`);
                    if (frame) {
                        // Transform empty state to sketch state
                        if (panelId === 1) {
                            frame.innerHTML = `
                                <div class="sketch-generated-state">
                                    <div class="sketch-placeholder-img" style="background: linear-gradient(135deg, #222, #888);"></div>
                                    <div class="sketch-badge">SKETCH GENERATED</div>
                                </div>
                            `;
                        } else if (panelId === 2) {
                            // Show already formatted sketch state
                            frame.style.display = "block";
                        }
                    }
                    addConsoleLog(`[STORYBOARD] Panel ${panelId} successfully compiled!`);
                    if (callback) callback();
                }, 1000);
            }, 800);
        }, initialDelay);
    }

    // Regenerate Outline
    regenerateOutlineBtn.addEventListener("click", () => {
        addConsoleLog(`[SYSTEM] Recalculating outline beats...`);
        // Simple visual jitter to show it recalculated
        document.querySelectorAll(".panel-card textarea").forEach(area => {
            area.style.backgroundColor = "#FFF1DF";
            setTimeout(() => {
                area.style.backgroundColor = "var(--color-bg-base)";
            }, 300);
        });
        setTimeout(() => {
            addConsoleLog(`[SYSTEM] Outline beat structure optimized.`);
        }, 600);
    });

    // ---------------------------------------------------------
    // EXPORT DOWNLOAD SIMULATION
    // ---------------------------------------------------------
    exportActionBtns.forEach(btn => {
        btn.addEventListener("click", () => {
            const format = btn.getAttribute("data-format");
            const originalText = btn.innerHTML;
            
            // Set mock loading state
            btn.disabled = true;
            btn.innerHTML = `COMPILING ${format}... <span class="status-dot"></span>`;
            
            addConsoleLog(`[EXPORT] Initiating bundle compilation for format: ${format}`);
            
            setTimeout(() => {
                btn.disabled = false;
                btn.innerHTML = originalText;
                addConsoleLog(`[EXPORT] Completed download for: panel_scrap_comic.${format.toLowerCase()}`);
                
                // Show completion banner/alert
                alert(`Successfully generated and downloaded panel_scrap_comic.${format.toLowerCase()}!`);
            }, 1800);
        });
    });

    // ---------------------------------------------------------
    // PUBLISH PROJECT SUCCESS OVERLAY & CONFETTI
    // ---------------------------------------------------------
    publishProjectBtn.addEventListener("click", () => {
        successModal.style.display = "flex";
        addConsoleLog(`[SYSTEM] Comic project "Episode 01: The Awakening" officially published!`);
        startConfetti();
    });

    successCloseBtn.addEventListener("click", () => {
        successModal.style.display = "none";
        stopConfetti();
    });

    // ---------------------------------------------------------
    // CONSOLE LOGGER UTILITIES
    // ---------------------------------------------------------
    function addConsoleLog(message) {
        const timestamp = new Date().toTimeString().split(' ')[0];
        const logLine = document.createElement("div");
        logLine.className = "log-line";
        
        if (message.startsWith("[SYSTEM]")) {
            logLine.innerHTML = `<span class="text-gray">[${timestamp}]</span> <span class="text-yellow">${message}</span>`;
        } else {
            logLine.innerHTML = `<span class="text-gray">[${timestamp}]</span> ${message}`;
        }
        
        consoleLogsContainer.appendChild(logLine);
        
        // Auto scroll to bottom
        consoleLogsContainer.scrollTop = consoleLogsContainer.scrollHeight;
    }

    // ---------------------------------------------------------
    // CONFETTI ANIMATION (PURE JS CANVAS SHARDS)
    // ---------------------------------------------------------
    let canvasConfetti = null;
    let ctxConfetti = null;
    let confettiActive = false;
    let confettiParticles = [];

    function startConfetti() {
        // Create full screen background canvas for particles
        if (!canvasConfetti) {
            canvasConfetti = document.createElement("canvas");
            canvasConfetti.style.position = "absolute";
            canvasConfetti.style.top = "0";
            canvasConfetti.style.left = "0";
            canvasConfetti.style.width = "100%";
            canvasConfetti.style.height = "100%";
            canvasConfetti.style.pointerEvents = "none";
            canvasConfetti.style.zIndex = "999";
            document.querySelector(".app-container").appendChild(canvasConfetti);
            ctxConfetti = canvasConfetti.getContext("2d");
        }
        
        canvasConfetti.width = canvasConfetti.offsetWidth;
        canvasConfetti.height = canvasConfetti.offsetHeight;
        confettiParticles = [];
        
        // Generate particles
        for (let i = 0; i < 60; i++) {
            confettiParticles.push({
                x: Math.random() * canvasConfetti.width,
                y: Math.random() * -100,
                color: ["#FAF6EE", "#F4B942", "#C84B31", "#1A1A1A"][Math.floor(Math.random() * 4)],
                size: Math.random() * 8 + 4,
                speedY: Math.random() * 4 + 2,
                speedX: Math.random() * 3 - 1.5,
                rot: Math.random() * 360,
                rotSpeed: Math.random() * 4 - 2
            });
        }
        
        confettiActive = true;
        animateConfetti();
    }

    function animateConfetti() {
        if (!confettiActive) return;
        ctxConfetti.clearRect(0, 0, canvasConfetti.width, canvasConfetti.height);
        
        let activeParticles = 0;
        confettiParticles.forEach(p => {
            p.y += p.speedY;
            p.x += p.speedX;
            p.rot += p.rotSpeed;
            
            if (p.y < canvasConfetti.height) {
                activeParticles++;
                ctxConfetti.save();
                ctxConfetti.translate(p.x, p.y);
                ctxConfetti.rotate((p.rot * Math.PI) / 180);
                
                ctxConfetti.fillStyle = p.color;
                ctxConfetti.lineWidth = 1.5;
                ctxConfetti.strokeStyle = "var(--color-border)";
                
                // Draw square confetti with brutalist thick outline
                ctxConfetti.fillRect(-p.size / 2, -p.size / 2, p.size, p.size);
                ctxConfetti.strokeRect(-p.size / 2, -p.size / 2, p.size, p.size);
                
                ctxConfetti.restore();
            }
        });
        
        if (activeParticles > 0) {
            requestAnimationFrame(animateConfetti);
        } else {
            stopConfetti();
        }
    }

    function stopConfetti() {
        confettiActive = false;
        if (canvasConfetti && canvasConfetti.parentElement) {
            canvasConfetti.parentElement.removeChild(canvasConfetti);
            canvasConfetti = null;
            ctxConfetti = null;
        }
    }

});
