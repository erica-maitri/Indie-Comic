/* -------------------------------------------------------------
   PANEL SCRAP - FRONTEND LOGIC & PIPELINE INTEGRATION
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
    const engineLlm = document.getElementById("engine-llm");
    const engineTemplate = document.getElementById("engine-template");
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
    const consoleStatus = document.getElementById("console-status");
    const emotionCanvas = document.getElementById("emotion-canvas");
    const timelineContainer = document.getElementById("timeline-container");
    const timelineEpisodeTitle = document.getElementById("timeline-episode-title");
    
    // Export & Publish Controls
    const pagePreviewLabel = document.getElementById("page-preview-label");
    const pagePreviewWrapper = document.getElementById("page-preview-wrapper");
    const exportActionBtns = document.querySelectorAll(".export-action-btn");
    const publishProjectBtn = document.getElementById("publish-project-btn");
    const successModal = document.getElementById("success-modal");
    const successCloseBtn = document.getElementById("success-close-btn");

    // Global variables to store state
    let activeStoryContext = null;
    let generatedOutlinePanels = [];
    let compiledFilePaths = {
        pdf: "",
        cbz: "",
        html: ""
    };
    let graphPoints = [];
    let selectedPointIndex = -1;

    function getAdvancedPayload() {
        const activeResBtn = document.querySelector(".toggle-group-sm button[id^='res-'].active");
        const activeStepsBtn = document.querySelector(".toggle-group-sm button[id^='steps-'].active");
        
        return {
            heat_alpha: parseFloat(document.getElementById("slider-heat-alpha").value),
            attention_blend: parseFloat(document.getElementById("slider-attention-blend").value),
            spatial_strength: parseFloat(document.getElementById("slider-spatial-strength").value),
            critic_threshold: parseFloat(document.getElementById("slider-critic-threshold").value),
            
            width: activeResBtn ? parseInt(activeResBtn.getAttribute("data-val")) : 768,
            height: activeResBtn ? parseInt(activeResBtn.getAttribute("data-val")) : 768,
            inference_steps: activeStepsBtn ? parseInt(activeStepsBtn.getAttribute("data-val")) : 25,
            lora_scale: parseFloat(document.getElementById("slider-lora-scale").value),
            seed: parseInt(document.getElementById("generation-seed-input").value) || 42,
            
            enable_ipadapter: document.getElementById("check-ipadapter").checked,
            enable_controlnet: document.getElementById("check-controlnet").checked,
            enable_cpuoffload: document.getElementById("check-cpuoffload").checked,
            
            enable_ssim: document.getElementById("metric-ssim").checked,
            enable_edge: document.getElementById("metric-edge").checked,
            enable_color: document.getElementById("metric-color").checked,
            enable_style: document.getElementById("metric-style").checked,
            enable_clip: document.getElementById("metric-clip").checked,
            enable_dinov2: document.getElementById("metric-dinov2").checked,
        };
    }

    // ---------------------------------------------------------
    // TAB NAVIGATION
    // ---------------------------------------------------------
    function switchTab(viewId) {
        [tabWriter, tabStoryboard, tabExport].forEach(tab => tab.classList.remove("active"));
        [writerView, storyboardView, exportView].forEach(view => view.classList.remove("active"));
        
        if (viewId === "writer") {
            tabWriter.classList.add("active");
            writerView.classList.add("active");
        } else if (viewId === "storyboard") {
            tabStoryboard.classList.add("active");
            storyboardView.classList.add("active");
            setTimeout(initEmotionArc, 50);
        } else if (viewId === "export") {
            tabExport.classList.add("active");
            exportView.classList.add("active");
        }
        
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
    // STORY ENGINE SOURCE TOGGLE
    // ---------------------------------------------------------
    [engineLlm, engineTemplate].forEach(btn => {
        btn.addEventListener("click", () => {
            engineLlm.classList.remove("active");
            engineTemplate.classList.remove("active");
            btn.classList.add("active");
            addConsoleLog(`[SYSTEM] Switched story engine to: ${btn.textContent.trim()}`);
        });
    });

    // ---------------------------------------------------------
    // CROSSOVER ENGINE TOGGLE
    // ---------------------------------------------------------
    const crossoverEnable = document.getElementById("crossover-enable");
    const crossoverDisable = document.getElementById("crossover-disable");
    [crossoverEnable, crossoverDisable].forEach(btn => {
        btn.addEventListener("click", () => {
            crossoverEnable.classList.remove("active");
            crossoverDisable.classList.remove("active");
            btn.classList.add("active");
            addConsoleLog(`[SYSTEM] Switched crossover engine to: ${btn.textContent.trim()}`);
        });
    });

    // ---------------------------------------------------------
    // ADVANCED TUNING SETTINGS COLLAPSIBLE
    // ---------------------------------------------------------
    const advancedToggleBtn = document.getElementById("advanced-toggle-btn");
    const advancedSettingsContent = document.getElementById("advanced-settings-content");
    advancedToggleBtn.addEventListener("click", () => {
        const isHidden = advancedSettingsContent.style.display === "none";
        advancedSettingsContent.style.display = isHidden ? "block" : "none";
        advancedToggleBtn.classList.toggle("open", isHidden);
    });

    // Update value badges for sliders
    const sliderHeatAlpha = document.getElementById("slider-heat-alpha");
    const heatAlphaVal = document.getElementById("heat-alpha-val");
    sliderHeatAlpha.addEventListener("input", (e) => {
        heatAlphaVal.textContent = e.target.value;
    });

    const sliderAttentionBlend = document.getElementById("slider-attention-blend");
    const attentionBlendVal = document.getElementById("attention-blend-val");
    sliderAttentionBlend.addEventListener("input", (e) => {
        attentionBlendVal.textContent = e.target.value;
    });

    const sliderSpatialStrength = document.getElementById("slider-spatial-strength");
    const spatialStrengthVal = document.getElementById("spatial-strength-val");
    sliderSpatialStrength.addEventListener("input", (e) => {
        spatialStrengthVal.textContent = e.target.value;
    });

    const sliderCriticThreshold = document.getElementById("slider-critic-threshold");
    const criticThresholdVal = document.getElementById("critic-threshold-val");
    sliderCriticThreshold.addEventListener("input", (e) => {
        criticThresholdVal.textContent = e.target.value;
    });

    const sliderLoraScale = document.getElementById("slider-lora-scale");
    const loraScaleVal = document.getElementById("lora-scale-val");
    sliderLoraScale.addEventListener("input", (e) => {
        loraScaleVal.textContent = e.target.value;
    });

    // Preset Toggles
    const resBtns = [document.getElementById("res-draft"), document.getElementById("res-normal"), document.getElementById("res-high")];
    resBtns.forEach(btn => {
        btn.addEventListener("click", () => {
            resBtns.forEach(b => b.classList.remove("active"));
            btn.classList.add("active");
            addConsoleLog(`[SYSTEM] Switched resolution preset to: ${btn.textContent.trim()}`);
        });
    });

    const stepsBtns = [document.getElementById("steps-draft"), document.getElementById("steps-normal"), document.getElementById("steps-high")];
    stepsBtns.forEach(btn => {
        btn.addEventListener("click", () => {
            stepsBtns.forEach(b => b.classList.remove("active"));
            btn.classList.add("active");
            addConsoleLog(`[SYSTEM] Switched inference steps preset to: ${btn.textContent.trim()}`);
        });
    });

    // ---------------------------------------------------------
    // SINGLE PANEL REGENERATION EVENT DELEGATION
    // ---------------------------------------------------------
    timelineContainer.addEventListener("click", async (e) => {
        const regenBtn = e.target.closest(".regen-panel-btn");
        if (!regenBtn) return;
        
        const panelId = parseInt(regenBtn.getAttribute("data-panel-id"));
        
        // Show loading state for this panel's image frame
        const frame = document.getElementById(`panel-img-frame-${panelId}`);
        frame.innerHTML = `
            <div class="empty-frame-state pulse">
                <span class="status-dot"></span>
                <span>REGENERATING PANEL ${panelId}...</span>
            </div>
        `;
        
        regenBtn.disabled = true;
        regenBtn.textContent = "REGENERATING...";
        
        try {
            // Collect all panels current data
            const panelCards = document.querySelectorAll(".panel-card");
            const panelsPayload = [];
            panelCards.forEach(card => {
                const pid = parseInt(card.getAttribute("data-panel-id"));
                const visualDescription = card.querySelector(".panel-desc-input").value.trim();
                const emotion = card.querySelector(".emotion-select").value;
                const dialogue = card.querySelector(".panel-dialogue-input").value.trim();
                
                panelsPayload.push({
                    panel_id: pid,
                    visual_description: visualDescription,
                    emotion: emotion,
                    dialogue: dialogue
                });
            });

            const payload = {
                panel_id: panelId,
                premise: premiseInput.value.trim(),
                panel_count: panelsPayload.length,
                style: selectedStyle,
                custom_style: customStylePromptText,
                character_name: document.getElementById("character-name-input").value.trim() || (activeStoryContext ? activeStoryContext.character_name : "Wanderer"),
                character_characteristics: document.getElementById("character-desc-input").value.trim() || null,
                story_world: document.getElementById("story-world-input").value.trim() || (activeStoryContext ? activeStoryContext.story_world : "The Abstract"),
                emotion: activeStoryContext ? activeStoryContext.emotion : "sadness",
                panels: panelsPayload,
                ...getAdvancedPayload()
            };

            const response = await fetch("/api/regenerate_panel", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload)
            });

            if (!response.ok) {
                const errData = await response.json();
                throw new Error(errData.error || "Failed to regenerate panel.");
            }

            const data = await response.json();
            
            // Update the panel image
            frame.innerHTML = `
                <div class="sketch-generated-state">
                    <img src="${data.image_path}?t=${Date.now()}" class="panel-result-image" alt="Panel ${panelId}">
                    <div class="sketch-badge">RENDERED</div>
                </div>
            `;
            
            // Re-render layout preview on the Export page
            renderPageReviewGrid(data.pages);
            
            addConsoleLog(`[SYSTEM] Panel ${panelId} successfully regenerated and page layouts updated!`);
            
        } catch (err) {
            addConsoleLog(`[ERROR] Failed to regenerate panel ${panelId}: ${err.message}`);
            alert(`Regeneration Error: ${err.message}`);
            // Revert back
            frame.innerHTML = `
                <div class="empty-frame-state">
                    <span>FAILED TO RENDER. RETRY.</span>
                </div>
            `;
        } finally {
            regenBtn.disabled = false;
            regenBtn.textContent = "REGENERATE PANEL";
        }
    });

    // ---------------------------------------------------------
    // ART STYLE SELECTION & MODAL
    // ---------------------------------------------------------
    let selectedStyle = "noir";
    let customStylePromptText = "";

    styleCards.forEach(card => {
        card.addEventListener("click", () => {
            if (card.id === "style-custom-btn") {
                styleModal.style.display = "flex";
                return;
            }
            
            styleCards.forEach(c => c.classList.remove("selected"));
            card.classList.add("selected");
            selectedStyle = card.getAttribute("data-style");
            
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
            customStylePromptText = promptText;
            selectedStyle = "custom";
            
            const customBadge = styleCustomBtn.querySelector(".style-badge");
            customBadge.textContent = promptText.substring(0, 10).toUpperCase() + "...";
            
            styleCards.forEach(c => c.classList.remove("selected"));
            styleCustomBtn.classList.add("selected");
            
            addConsoleLog(`[SYSTEM] Custom style prompt set: "${promptText}"`);
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
    // STORY OUTLINE GENERATION (BACKEND INTEGRATION)
    // ---------------------------------------------------------
    generateOutlineBtn.addEventListener("click", async () => {
        const premise = premiseInput.value.trim();
        if (!premise) {
            alert("Please enter a story premise first!");
            return;
        }

        // Show loading state
        generateOutlineBtn.parentElement.style.display = "none";
        generationLoader.style.display = "block";
        loaderProgress.style.width = "0%";
        loaderTimeLeft.textContent = "15";
        loaderStatusText.textContent = "Analyzing premise...";

        // Simple progress simulation while waiting for API response
        let fakeProgress = 0;
        const progressInterval = setInterval(() => {
            if (fakeProgress < 90) {
                fakeProgress += 5;
                loaderProgress.style.width = `${fakeProgress}%`;
                loaderTimeLeft.textContent = Math.max(1, Math.round((100 - fakeProgress) * 0.15)) + "s";
                
                if (fakeProgress > 70) {
                    loaderStatusText.textContent = "Assembling story beats...";
                } else if (fakeProgress > 40) {
                    loaderStatusText.textContent = "Running emotion analyzer...";
                }
            }
        }, 300);

        try {
            const payload = {
                premise: premise,
                panel_count: parseInt(panelSlider.value),
                mode: modeQuick.classList.contains("active") ? "quick" : "guided",
                engine: engineLlm.classList.contains("active") ? "llm" : "template",
                style: selectedStyle,
                custom_style: customStylePromptText,
                character_name: document.getElementById("character-name-input").value.trim() || null,
                character_characteristics: document.getElementById("character-desc-input").value.trim() || null,
                story_world: document.getElementById("story-world-input").value.trim() || null,
                weave_mood: crossoverEnable.classList.contains("active"),
                ...getAdvancedPayload()
            };

            const response = await fetch("/api/generate_outline", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload)
            });

            const data = await response.json();
            clearInterval(progressInterval);

            if (!response.ok) {
                throw new Error(data.error || "Failed to generate outline.");
            }

            // Save story context and panels
            activeStoryContext = data;
            generatedOutlinePanels = data.panels;
            
            // Update UI components with backend results
            timelineEpisodeTitle.textContent = `EPISODE: ${data.character_name.toUpperCase()} IN ${data.story_world.toUpperCase()}`;
            
            // Clear and render panel cards dynamically
            renderTimelinePanels(data.panels);
            
            // Map graph control points based on returned intensities
            graphPoints = data.intensity_points.map((pt, idx) => {
                const totalPts = data.intensity_points.length;
                const xVal = 30 + idx * (250 / (totalPts - 1));
                return {
                    x: xVal,
                    y: pt.y,
                    label: pt.label,
                    draggable: (idx > 0 && idx < totalPts - 1)
                };
            });

            // Finish loader
            loaderProgress.style.width = "100%";
            loaderTimeLeft.textContent = "0";
            loaderStatusText.textContent = "Outline ready!";

            setTimeout(() => {
                generationLoader.style.display = "none";
                generateOutlineBtn.parentElement.style.display = "block";
                switchTab("storyboard");
                addConsoleLog(`[STORY] Successfully generated ${data.panels.length}-panel outline matching routed emotion '${data.emotion.toUpperCase()}'!`);
            }, 600);

        } catch (err) {
            clearInterval(progressInterval);
            generationLoader.style.display = "none";
            generateOutlineBtn.parentElement.style.display = "block";
            alert(`Error generating story: ${err.message}`);
            console.error(err);
        }
    });

    // Dynamically render timeline cards
    function renderTimelinePanels(panels) {
        // Clear previous children except the line itself
        const timelineLine = timelineContainer.querySelector(".timeline-line");
        timelineContainer.innerHTML = "";
        timelineContainer.appendChild(timelineLine);

        panels.forEach((p, idx) => {
            const item = document.createElement("div");
            item.className = "timeline-item";
            item.innerHTML = `
                <div class="timeline-badge">${p.panel_id}</div>
                <div class="brutalist-card panel-card" data-panel-id="${p.panel_id}">
                    <div class="panel-field-group">
                        <div class="field-label">Visual Description</div>
                        <textarea class="panel-desc-input">${p.visual_description}</textarea>
                    </div>
                    <div class="panel-field-group emotion-row">
                        <div>
                            <div class="field-label">Character Emotion</div>
                            <div class="select-wrapper">
                                <select class="emotion-select">
                                    <option value="AWE" ${p.emotion === 'AWE' ? 'selected' : ''}>AWE</option>
                                    <option value="DETERMINATION" ${p.emotion === 'DETERMINATION' ? 'selected' : ''}>DETERMINATION</option>
                                    <option value="JOY" ${p.emotion === 'JOY' ? 'selected' : ''}>JOY</option>
                                    <option value="SADNESS" ${p.emotion === 'SADNESS' ? 'selected' : ''}>SADNESS</option>
                                    <option value="ANGER" ${p.emotion === 'ANGER' ? 'selected' : ''}>ANGER</option>
                                    <option value="FEAR" ${p.emotion === 'FEAR' ? 'selected' : ''}>FEAR</option>
                                    <option value="NEUTRAL" ${p.emotion === 'NEUTRAL' || !p.emotion ? 'selected' : ''}>NEUTRAL</option>
                                </select>
                            </div>
                        </div>
                        <div class="regen-badge-slot">
                            <span class="badge-gray">PANEL ${p.panel_id}</span>
                        </div>
                    </div>
                    <div class="panel-field-group">
                        <div class="field-label">Dialogue</div>
                        <textarea class="panel-dialogue-input">${p.dialogue}</textarea>
                    </div>
                    <div class="panel-image-container">
                        <div class="empty-frame-state" id="panel-img-frame-${p.panel_id}">
                            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                                <rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect>
                                <circle cx="8.5" cy="8.5" r="1.5"></circle>
                                <polyline points="21 15 16 10 5 21"></polyline>
                            </svg>
                            <span>EMPTY FRAME</span>
                        </div>
                        <button class="brutalist-btn btn-yellow btn-sm regen-panel-btn" id="regen-panel-btn-${p.panel_id}" data-panel-id="${p.panel_id}" style="margin-top: 10px; font-size: 12px; padding: 6px 12px; display: none;">REGENERATE PANEL</button>
                    </div>
                </div>
            `;
            timelineContainer.appendChild(item);
        });
    }

    // ---------------------------------------------------------
    // STORYBOARD: INTERACTIVE EMOTION ARC GRAPH (HTML5 CANVAS)
    // ---------------------------------------------------------
    function initEmotionArc() {
        if (!emotionCanvas) return;
        const ctx = emotionCanvas.getContext("2d");
        
        // If graphPoints is empty, define default fallback points
        if (graphPoints.length === 0) {
            graphPoints = [
                { x: 30, y: 100, label: "Start", draggable: false },
                { x: 120, y: 70, label: "Build-up", draggable: true },
                { x: 180, y: 40, label: "Climax", draggable: true },
                { x: 230, y: 80, label: "Fall", draggable: true },
                { x: 280, y: 110, label: "Ending", draggable: false }
            ];
        }
        
        drawChart();
        
        emotionCanvas.onmousedown = (e) => {
            const rect = emotionCanvas.getBoundingClientRect();
            const scaleX = emotionCanvas.width / rect.width;
            const scaleY = emotionCanvas.height / rect.height;
            const mouseX = (e.clientX - rect.left) * scaleX;
            const mouseY = (e.clientY - rect.top) * scaleY;
            
            selectedPointIndex = graphPoints.findIndex(pt => {
                if (!pt.draggable) return false;
                const dist = Math.sqrt(Math.pow(pt.x - mouseX, 2) + Math.pow(pt.y - mouseY, 2));
                return dist < 12;
            });
            
            if (selectedPointIndex !== -1) {
                addConsoleLog(`[UI] Adjusting climax beat intensity at: ${graphPoints[selectedPointIndex].label}`);
            }
        };

        emotionCanvas.onmousemove = (e) => {
            if (selectedPointIndex === -1) return;
            const rect = emotionCanvas.getBoundingClientRect();
            const scaleY = emotionCanvas.height / rect.height;
            const mouseY = (e.clientY - rect.top) * scaleY;
            
            graphPoints[selectedPointIndex].y = Math.max(15, Math.min(mouseY, 115));
            drawChart();
        };

        emotionCanvas.onmouseup = () => {
            if (selectedPointIndex !== -1) {
                const intensityVal = Math.round(((130 - graphPoints[selectedPointIndex].y) / 130) * 100);
                addConsoleLog(`[SYSTEM] Adjusted beat "${graphPoints[selectedPointIndex].label}" climax intensity set to ${intensityVal}%`);
                selectedPointIndex = -1;
            }
        };
        
        // Touch events for mobile compatibility
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
            
            for (let i = 0; i < graphPoints.length - 1; i++) {
                const xc = (graphPoints[i].x + graphPoints[i+1].x) / 2;
                const yc = (graphPoints[i].y + graphPoints[i+1].y) / 2;
                ctx.quadraticCurveTo(graphPoints[i].x, graphPoints[i].y, xc, yc);
            }
            ctx.lineTo(graphPoints[graphPoints.length - 1].x, graphPoints[graphPoints.length - 1].y);
            
            ctx.strokeStyle = "var(--color-accent-red)";
            ctx.lineWidth = 4;
            ctx.stroke();
            
            // Draw control point dots
            graphPoints.forEach(pt => {
                ctx.beginPath();
                ctx.arc(pt.x, pt.y, pt.draggable ? 6 : 4, 0, 2 * Math.PI);
                ctx.fillStyle = pt.draggable ? "#7A2012" : "var(--color-border)";
                ctx.fill();
                
                if (pt.draggable) {
                    ctx.strokeStyle = "var(--color-accent-yellow)";
                    ctx.lineWidth = 2.5;
                    ctx.stroke();
                }
            });
        }
    }

    // ---------------------------------------------------------
    // STORYBOARD: DRAW COMIC PANELS (SSE LOG STREAMING)
    // ---------------------------------------------------------
    drawPanelsBtn.addEventListener("click", async () => {
        // Collect current panel inputs from HTML
        const panelCards = document.querySelectorAll(".panel-card");
        const panelsPayload = [];
        panelCards.forEach(card => {
            const pid = parseInt(card.getAttribute("data-panel-id"));
            const visualDescription = card.querySelector(".panel-desc-input").value.trim();
            const emotion = card.querySelector(".emotion-select").value;
            const dialogue = card.querySelector(".panel-dialogue-input").value.trim();
            
            panelsPayload.push({
                panel_id: pid,
                visual_description: visualDescription,
                emotion: emotion,
                dialogue: dialogue
            });
        });

        if (panelsPayload.length === 0) {
            alert("No panels found. Please generate a story outline first!");
            return;
        }

        // Set loader states
        consoleStatus.textContent = "RENDERING...";
        consoleStatus.style.color = "var(--color-accent-red)";
        drawPanelsBtn.disabled = true;
        drawPanelsBtn.innerHTML = `RENDERING... <span class="status-dot"></span>`;
        
        // Clear log box
        consoleLogsContainer.innerHTML = "";
        addConsoleLog("[SYSTEM] Initiating backend execution pipeline...");
        
        // Scroll to terminal console view
        document.querySelector(".system-log-console").scrollIntoView({ behavior: 'smooth' });

        try {
            const payload = {
                premise: premiseInput.value.trim(),
                panel_count: panelsPayload.length,
                style: selectedStyle,
                custom_style: customStylePromptText,
                character_name: document.getElementById("character-name-input").value.trim() || (activeStoryContext ? activeStoryContext.character_name : "Wanderer"),
                character_characteristics: document.getElementById("character-desc-input").value.trim() || null,
                story_world: document.getElementById("story-world-input").value.trim() || (activeStoryContext ? activeStoryContext.story_world : "The Abstract"),
                emotion: activeStoryContext ? activeStoryContext.emotion : "sadness",
                weave_mood: crossoverEnable.classList.contains("active"),
                panels: panelsPayload,
                ...getAdvancedPayload()
            };

            const response = await fetch("/api/draw_panels", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload)
            });

            if (!response.ok) {
                const data = await response.json();
                throw new Error(data.error || "Failed to start pipeline.");
            }

            // Stream logs via SSE parser (Response Reader)
            const reader = response.body.getReader();
            const decoder = new TextDecoder("utf-8");
            let buffer = "";

            while (true) {
                const { value, done } = await reader.read();
                if (done) break;

                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split("\n");
                // The last element is a partial line, keep it in the buffer
                buffer = lines.pop();

                for (const line of lines) {
                    if (line.startsWith("data: ")) {
                        const jsonStr = line.substring(6).trim();
                        if (!jsonStr) continue;

                        try {
                            const data = JSON.parse(jsonStr);
                            
                            if (data.log) {
                                addConsoleLog(data.log);
                            }
                            
                            if (data.status === "complete") {
                                handleRenderingComplete(data);
                            } else if (data.status === "error") {
                                throw new Error(data.error);
                            }
                        } catch (e) {
                            console.error("Error parsing log line", line, e);
                        }
                    }
                }
            }

        } catch (err) {
            consoleStatus.textContent = "ERROR";
            consoleStatus.style.color = "var(--color-accent-red)";
            addConsoleLog(`[ERROR] Pipeline crashed: ${err.message}`);
            alert(`Pipeline Error: ${err.message}`);
            drawPanelsBtn.disabled = false;
            drawPanelsBtn.innerHTML = `
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
                    <path d="M12 20h9"></path>
                    <path d="M16.5 3.5a2.121 2.121 0 0 1 3 3L7 19l-4 1 1-4L16.5 3.5z"></path>
                </svg>
                DRAW COMIC PANELS
            `;
        }
    });

    // Handle completed pipeline drawing run
    function handleRenderingComplete(data) {
        consoleStatus.textContent = "READY";
        consoleStatus.style.color = "var(--color-text-dark)";
        drawPanelsBtn.disabled = false;
        drawPanelsBtn.innerHTML = `
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
                <path d="M12 20h9"></path>
                <path d="M16.5 3.5a2.121 2.121 0 0 1 3 3L7 19l-4 1 1-4L16.5 3.5z"></path>
            </svg>
            DRAW COMIC PANELS
        `;

        addConsoleLog(`[SYSTEM] Generation and layout assembly complete! Updating UI...`);

        // 1. Update timeline panel images in storyboard view
        data.panels.forEach(p => {
            const frame = document.getElementById(`panel-img-frame-${p.panel_id}`);
            if (frame) {
                frame.innerHTML = `
                    <div class="sketch-generated-state">
                        <img src="${p.image_path}" class="panel-result-image" alt="Panel ${p.panel_id}">
                        <div class="sketch-badge">RENDERED</div>
                    </div>
                `;
            }
        });

        // Show all regenerate buttons
        document.querySelectorAll(".regen-panel-btn").forEach(btn => {
            btn.style.display = "block";
        });

        // 2. Cache output files
        compiledFilePaths.pdf = data.pdf_path;
        compiledFilePaths.cbz = data.cbz_path;
        compiledFilePaths.html = data.html_path;

        // 3. Render Export layout preview page
        renderPageReviewGrid(data.pages);

        // 4. Update stats values
        document.querySelector(".stats-value:nth-child(2)").textContent = data.panels.length;
        
        // Auto-navigate to Export review tab
        setTimeout(() => {
            switchTab("export");
        }, 1500);
    }

    // Render MangaFlow Page Layout Preview Grid
    function renderPageReviewGrid(pages) {
        pagePreviewWrapper.innerHTML = "";

        if (pages.length === 0) {
            pagePreviewWrapper.innerHTML = `
                <div class="empty-layout-state">
                    <span>Layout file not found. Render again.</span>
                </div>
            `;
            return;
        }

        // Render page selector tabs if multiple pages are returned
        const reviewCard = document.querySelector(".page-review-card");
        let tabsContainer = document.getElementById("page-review-tabs");
        
        if (!tabsContainer) {
            tabsContainer = document.createElement("div");
            tabsContainer.id = "page-review-tabs";
            tabsContainer.className = "page-review-tabs";
            // Insert tabs container before the wrapper
            reviewCard.insertBefore(tabsContainer, pagePreviewWrapper);
        }
        
        tabsContainer.innerHTML = "";
        
        pages.forEach((page, index) => {
            const btn = document.createElement("button");
            btn.className = `page-tab-btn ${index === 0 ? 'active' : ''}`;
            btn.textContent = `PAGE ${page.page_num}`;
            btn.addEventListener("click", () => {
                document.querySelectorAll(".page-tab-btn").forEach(b => b.classList.remove("active"));
                btn.classList.add("active");
                pagePreviewLabel.textContent = `PAGE ${page.page_num} LAYOUT`;
                
                // Switch image source
                const previewImg = pagePreviewWrapper.querySelector("img");
                if (previewImg) previewImg.src = page.image_path;
            });
            tabsContainer.appendChild(btn);
        });

        // Show the first page layout by default
        pagePreviewLabel.textContent = `PAGE ${pages[0].page_num} LAYOUT`;
        const img = document.createElement("img");
        img.src = pages[0].image_path;
        img.className = "assembled-layout-preview";
        img.alt = `Page ${pages[0].page_num} Layout`;
        pagePreviewWrapper.appendChild(img);
    }

    // ---------------------------------------------------------
    // EXPORT DOWNLOADS & ACTIONS
    // ---------------------------------------------------------
    exportActionBtns.forEach(btn => {
        btn.addEventListener("click", () => {
            const format = btn.getAttribute("data-format").toLowerCase();
            const filePath = compiledFilePaths[format];
            
            if (!filePath) {
                alert(`No generated ${format.toUpperCase()} file found. Make sure to generate panels first!`);
                return;
            }

            const originalText = btn.innerHTML;
            btn.disabled = true;
            btn.innerHTML = `DOWNLOADING... <span class="status-dot"></span>`;
            
            addConsoleLog(`[EXPORT] Requesting file download for format: ${format.toUpperCase()}`);
            
            // Redirect browser to secure download API route
            window.location.href = `/api/download?path=${encodeURIComponent(filePath)}`;

            setTimeout(() => {
                btn.disabled = false;
                btn.innerHTML = originalText;
                addConsoleLog(`[EXPORT] Completed file download response!`);
            }, 1000);
        });
    });

    // ---------------------------------------------------------
    // PUBLISH PROJECT OVERLAY
    // ---------------------------------------------------------
    publishProjectBtn.addEventListener("click", () => {
        successModal.style.display = "flex";
        addConsoleLog(`[SYSTEM] Comic project officially published successfully!`);
        startConfetti();
    });

    successCloseBtn.addEventListener("click", () => {
        successModal.style.display = "none";
        stopConfetti();
    });

    // Regenerate beats outline simulation trigger
    regenerateOutlineBtn.addEventListener("click", () => {
        addConsoleLog(`[SYSTEM] Recalculating story beat structures...`);
        document.querySelectorAll(".panel-card textarea").forEach(area => {
            area.style.backgroundColor = "#FFF1DF";
            setTimeout(() => {
                area.style.backgroundColor = "var(--color-bg-base)";
            }, 300);
        });
        setTimeout(() => {
            addConsoleLog(`[SYSTEM] Story beat pacing optimized.`);
        }, 500);
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
        } else if (message.includes("[ERROR]")) {
            logLine.innerHTML = `<span class="text-gray">[${timestamp}]</span> <span class="text-red">${message}</span>`;
        } else {
            logLine.innerHTML = `<span class="text-gray">[${timestamp}]</span> ${message}`;
        }
        
        consoleLogsContainer.appendChild(logLine);
        
        // Auto scroll console to bottom
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
