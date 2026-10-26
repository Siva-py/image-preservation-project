// ============================================================
// IMAGE PRESERVE — FRONTEND SCRIPT
// ============================================================


// ============================================================
// 1. GENERAL HELPERS AND DASHBOARD DATA
// ============================================================

const $ = (selector) => document.querySelector(selector);

let report = null;
let isPipelineRunning = false;


function formatNumber(value, decimals = 2) {
    return Number.isFinite(Number(value))
        ? Number(value).toFixed(decimals)
        : "—";
}


function toast(message) {
    const toastElement = $("#toast");

    toastElement.textContent = message;
    toastElement.classList.add("show");

    clearTimeout(window.toastTimeout);

    window.toastTimeout = setTimeout(() => {
        toastElement.classList.remove("show");
    }, 3000);
}


// ============================================================
// 2. PAGE NAVIGATION
// ============================================================

function showPage(pageId) {
    document.querySelectorAll(".page").forEach((element) => {
        element.classList.toggle(
            "active",
            element.id === pageId
        );
    });

    document.querySelectorAll("nav button").forEach((button) => {
        button.classList.toggle(
            "active",
            button.dataset.page === pageId
        );
    });

    const pageTitles = {
        home: "Preserve what matters.",
        process: "Process your images",
        results: "Your dataset, refined.",
        gallery: "Image gallery",
    };

    $("#page-title").textContent =
        pageTitles[pageId] || "ImagePreserve";

    window.scrollTo({
        top: 0,
        behavior: "smooth"
    });
}


document.querySelectorAll("[data-page]").forEach((element) => {
    element.addEventListener("click", () => {
        showPage(element.dataset.page === "analytics" ? "results" : element.dataset.page);
    });
});


const galleryTabs = [
    { button: document.getElementById("tab-original"), panel: document.getElementById("original-gallery") },
    { button: document.getElementById("tab-final"), panel: document.getElementById("final-gallery") },
];

galleryTabs.forEach(({ button, panel }) => {
    button.addEventListener("click", () => {
        galleryTabs.forEach(({ button: tabButton, panel: tabPanel }) => {
            const selected = tabButton === button;
            tabButton.classList.toggle("active", selected);
            tabButton.setAttribute("aria-selected", String(selected));
            tabPanel.classList.toggle("active", selected);
            tabPanel.hidden = !selected;
        });
    });
});

document.querySelectorAll("[data-go]").forEach((element) => {
    element.addEventListener("click", () => {
        showPage(element.dataset.go === "analytics" ? "results" : element.dataset.go);
    });
});


// ============================================================
// 3. COMPRESSION CONFIGURATION
// ============================================================

function updateCompressionOptions() {
    const enabled = $("#compression").checked;

    $("#compression-options").hidden = !enabled;
}

$("#compression").addEventListener(
    "change",
    updateCompressionOptions
);

updateCompressionOptions();


// Validate target image size as the user types.
$("#target-size-kb").addEventListener("input", (event) => {
    event.target.setCustomValidity("");
});


// Prevent negative values and scientific notation.
$("#target-size-kb").addEventListener("keydown", (event) => {
    if (
        event.key === "-" ||
        event.key === "e" ||
        event.key === "E"
    ) {
        event.preventDefault();
    }
});


// ============================================================
// 4. INFORMATION PRESERVATION CONFIGURATION
// ============================================================

$("#retention").addEventListener("input", (event) => {
    $("#retention-value").textContent =
        `${event.target.value}%`;
});


// ============================================================
// 5. HOME PAGE — 3D CIRCULAR IMAGE CAROUSEL
// ============================================================

// FastAPI server address.
// Change this if your backend uses another address or port.

const API_BASE_URL =
    window.IMAGEPRESERVE_API_BASE ||
    "http://127.0.0.1:8000";

const CAROUSEL_ENDPOINT = "/api/carousel-images";
const DATASET_IMAGES_ENDPOINT = "/api/dataset-images";
const FINAL_DATASET_DOWNLOAD_ENDPOINT = "/api/final-dataset/download";

let latestRunHasImages = false;

const SUPPORTED_IMAGE_EXTENSIONS = [
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".gif",
    ".avif",
    ".bmp"
];

// Time taken for one complete revolution.

const CAROUSEL_ROTATION_SECONDS = 60;

// State used by the carousel.

let carouselImages = [];
let normalCarouselImages = [];
let friendsCarouselImages = [];
let friendsModeActive = false;
let carouselRing = null;


// ------------------------------------------------------------
// 5.1 Check whether a filename is an image
// ------------------------------------------------------------

function isSupportedImage(filename) {
    if (typeof filename !== "string") {
        return false;
    }

    const lowercaseFilename = filename.toLowerCase();

    return SUPPORTED_IMAGE_EXTENSIONS.some((extension) => {
        return lowercaseFilename.endsWith(extension);
    });
}


// ------------------------------------------------------------
// 5.2 Create an image card
// ------------------------------------------------------------

function createCarouselCard(imageItem, index) {
    const card = document.createElement("figure");

    card.className = "carousel-card";
    card.dataset.index = String(index);

    const image = document.createElement("img");

    image.src = imageItem.src;
    image.alt = "";
    image.draggable = false;
    image.loading = index < 4 ? "eager" : "lazy";

    image.addEventListener("error", () => {
        card.classList.add("image-unavailable");
    });

    image.addEventListener("load", () => {
        card.classList.remove("image-unavailable");
    });

    card.appendChild(image);

    return card;
}


// ------------------------------------------------------------
// 5.3 Position images around the 3D ring
// ------------------------------------------------------------

function renderCarousel() {
    carouselRing = $("#carousel-slots");

    if (!carouselRing) {
        console.error(
            "Carousel container #carousel-slots was not found."
        );

        return;
    }

    carouselRing.replaceChildren();

    if (carouselImages.length === 0) {
        carouselRing.classList.add("is-empty");

        console.warn(
            "No carousel images were found. Check the image folder and API."
        );

        return;
    }

    const MAX_VISIBLE_CAROUSEL_IMAGES = 7;

    const visibleImages =
        carouselImages.length <= MAX_VISIBLE_CAROUSEL_IMAGES
            ? carouselImages
            : Array.from(
                { length: MAX_VISIBLE_CAROUSEL_IMAGES },
                (_, slot) => {
                    const index = Math.floor(
                        slot * carouselImages.length /
                        MAX_VISIBLE_CAROUSEL_IMAGES
                    );

                    return carouselImages[index];
                }
            );

    const angleBetweenImages = 360 / visibleImages.length;

    const ringRadius = Math.max(
        220,
        Math.min(290, visibleImages.length * 35)
    );

    carouselRing.style.setProperty(
        "--carousel-radius",
        `${ringRadius}px`
    );

    carouselRing.style.setProperty(
        "--carousel-duration",
        `${CAROUSEL_ROTATION_SECONDS}s`
    );

    visibleImages.forEach((imageItem, index) => {
        const card = createCarouselCard(imageItem, index);

        const angle = index * angleBetweenImages;

        card.style.transform = `
            translate(-50%, -50%)
            rotateY(${angle}deg)
            translateZ(${ringRadius}px)
        `;

        carouselRing.appendChild(card);
    });

    carouselRing.style.animation = "none";

    carouselRing.style.transform =
        `translateZ(-${ringRadius}px) rotateY(0deg)`;

    void carouselRing.offsetHeight;

    carouselRing.style.animation =
        `carousel-ring-rotation ${CAROUSEL_ROTATION_SECONDS}s linear infinite`;
}


// ------------------------------------------------------------
// 5.4 Load images from FastAPI
// ------------------------------------------------------------

async function loadCarouselImagesFromFolder() {
    try {
        const response = await fetch(
            `${API_BASE_URL}${CAROUSEL_ENDPOINT}`
        );

        if (!response.ok) {
            throw new Error(
                `Carousel API returned HTTP ${response.status}`
            );
        }

        const filenames = await response.json();

        if (!Array.isArray(filenames)) {
            throw new Error(
                "The carousel API did not return a list of filenames."
            );
        }

        carouselImages = filenames
            .filter(isSupportedImage)
            .map((filename) => ({
                src:
                    `${API_BASE_URL}${CAROUSEL_ENDPOINT}/` +
                    encodeURIComponent(filename)
            }));


        renderCarousel();

        console.log(
            `Loaded ${carouselImages.length} carousel images from FastAPI.`
        );

    } catch (error) {
        carouselImages = [];

        renderCarousel();

        console.error(
            "Could not load carousel images. Check that FastAPI is running, " +
            "the carousel router is registered, and the image folder exists.",
            error
        );
    }
}


// ------------------------------------------------------------
// 5.5 Secret friends carousel (click the logo to toggle)
// ------------------------------------------------------------

// FastAPI scans frontend/assets/friends and returns the current filenames.
// No friends.json file or manual filename list is required.
const FRIENDS_ENDPOINT = "/api/easter-egg/friends";
const FRIENDS_ASSET_PATH = "assets/friends";

async function loadFriendsCarouselImages() {
    try {
        const response = await fetch(
            `${API_BASE_URL}${FRIENDS_ENDPOINT}`,
            { cache: "no-store" }
        );

        if (!response.ok) {
            throw new Error(`Friends API returned HTTP ${response.status}`);
        }

        const data = await response.json();
        const filenames = Array.isArray(data) ? data : data.images;

        if (!Array.isArray(filenames)) {
            throw new Error(
                "The friends API response must contain an images array."
            );
        }

        friendsCarouselImages = filenames
            .filter((filename) =>
                typeof filename === "string" && isSupportedImage(filename)
            )
            .map((filename) => ({
                src: `${FRIENDS_ASSET_PATH}/${encodeURIComponent(filename)}`,
                filename
            }));

        console.log(
            `Loaded ${friendsCarouselImages.length} friends images from FastAPI.`
        );

        return friendsCarouselImages.length;
    } catch (error) {
        console.error(
            "Could not load friends images. Check that the FastAPI Easter egg " +
            "router is registered and frontend/assets/friends exists.",
            error
        );
        friendsCarouselImages = [];
        return 0;
    }
}

async function toggleFriendsCarousel(event) {
    if (event) {
        event.preventDefault();
        event.stopPropagation();
    }

    if (!friendsModeActive) {
        // Fetch the latest folder contents every time friends mode is activated.
        const friendsCount = await loadFriendsCarouselImages();

        if (friendsCount === 0) {
            showFriendsModeToast(
                "No supported photos found in assets/friends."
            );
            return;
        }

        normalCarouselImages = [...carouselImages];
        carouselImages = [...friendsCarouselImages];
        friendsModeActive = true;
        renderCarousel();
        showFriendsModeToast("🥚 Friends mode unlocked!");
    } else {
        friendsModeActive = false;
        carouselImages = [...normalCarouselImages];
        renderCarousel();
        showFriendsModeToast("Back to the ImagePreserve carousel.");
    }
}

function showFriendsModeToast(message) {
    let toast = document.getElementById("friends-mode-toast");
    if (!toast) {
        toast = document.createElement("div");
        toast.id = "friends-mode-toast";
        toast.className = "friends-mode-toast";
        toast.setAttribute("role", "status");
        document.body.appendChild(toast);
    }
    toast.textContent = message;
    toast.classList.add("show");
    window.clearTimeout(toast._hideTimer);
    toast._hideTimer = window.setTimeout(() => toast.classList.remove("show"), 2600);
}

const brandLogo = document.querySelector(".brand");
if (brandLogo) {
    brandLogo.addEventListener("click", toggleFriendsCarousel);
}

// ------------------------------------------------------------
// 5.6 Initialize carousel
// ------------------------------------------------------------

function initializeCarousel() {
    loadCarouselImagesFromFolder();
}

initializeCarousel();


// ============================================================
// 6. IMAGE UPLOAD AND PREVIEW
// ============================================================

let files = [];


$("#browse").addEventListener("click", () => {
    $("#files").click();
});


// Allow users to select a folder, including images in its subfolders.
// The browser provides these files through the webkitdirectory attribute.


$("#drop").addEventListener("click", (event) => {
    if (!event.target.closest("button")) {
        $("#files").click();
    }
});

$("#files").addEventListener("change", (event) => {
    addFiles([...event.target.files]);
    event.target.value = "";
});




["dragover", "dragenter"].forEach((eventName) => {
    $("#drop").addEventListener(eventName, (event) => {
        event.preventDefault();

        $("#drop").classList.add("drag");
    });
});


["dragleave", "drop"].forEach((eventName) => {
    $("#drop").addEventListener(eventName, (event) => {
        event.preventDefault();

        $("#drop").classList.remove("drag");
    });
});


$("#drop").addEventListener("drop", (event) => {
    const droppedImages = [...event.dataTransfer.files].filter((file) => {
        return file.type.startsWith("image/");
    });

    addFiles(droppedImages);
});


// ------------------------------------------------------------
// 6.1 Add uploaded files and refresh previews
// ------------------------------------------------------------

function addFiles(newFiles) {
    const acceptedFiles = [];
    const rejectedFiles = [];

    newFiles.forEach((file) => {
        const extension = file.name
            .slice(file.name.lastIndexOf("."))
            .toLowerCase();

        if ([".jpg", ".jpeg", ".png", ".webp"].includes(extension)) {
            acceptedFiles.push(file);
        } else {
            rejectedFiles.push(file.name);
        }
    });

    files.push(...acceptedFiles);

    updateFileSummary();
    renderFilePreviews();

    if (rejectedFiles.length > 0) {
        toast(
            `${rejectedFiles.length} unsupported file(s) skipped. ` +
            "Use JPG, JPEG, PNG, or WEBP."
        );
    } else if (acceptedFiles.length > 0) {
        toast(`${acceptedFiles.length} image(s) added to the upload queue.`);
    }
}


function updateFileSummary() {
    $("#file-count").textContent = files.length
        ? `${files.length} image(s) selected`
        : "No files selected";

    const totalSizeMB = files.reduce((total, file) => {
        return total + file.size;
    }, 0) / 1048576;

    $("#file-size").textContent = files.length
        ? `${totalSizeMB.toFixed(2)} MB · Ready to upload`
        : "Choose images to preview them here.";
}


function renderFilePreviews() {
    $("#previews").querySelectorAll("img").forEach((image) => {
        if (image.src.startsWith("blob:")) {
            URL.revokeObjectURL(image.src);
        }
    });

    $("#previews").replaceChildren();

    files.slice(0, 15).forEach((file) => {
        const preview = document.createElement("div");
        preview.className = "preview";

        const image = document.createElement("img");
        image.src = URL.createObjectURL(file);
        image.alt = file.name;

        const filename = document.createElement("span");
        filename.textContent = file.webkitRelativePath || file.name;
        filename.title = filename.textContent;

        preview.append(image, filename);
        $("#previews").appendChild(preview);
    });

    if (files.length > 15) {
        const remainingNote = document.createElement("p");
        remainingNote.className = "note";
        remainingNote.textContent =
            `Showing 15 previews. ${files.length - 15} more image(s) selected.`;

        $("#previews").appendChild(remainingNote);
    }
}


// ------------------------------------------------------------
// 6.2 Clear uploaded files
// ------------------------------------------------------------

$("#clear").addEventListener("click", () => {
    $("#previews").querySelectorAll("img").forEach((image) => {
        if (image.src.startsWith("blob:")) {
            URL.revokeObjectURL(image.src);
        }
    });

    files = [];
    $("#files").value = "";
    
    $("#previews").replaceChildren();

    updateFileSummary();
});


// ============================================================
// 7. PROCESSING CONFIGURATION PREVIEW
// ============================================================

$("#preview-run").addEventListener("click", runPipeline);


function validatePipelineSettings() {
    const targetSizeInput = $("#target-size-kb");
    const targetSizeKB = Number(targetSizeInput.value);

    targetSizeInput.setCustomValidity("");

    if (files.length === 0) {
        toast("Choose at least one image before running the pipeline.");
        showPage("process");
        return false;
    }

    if (
        $("#compression").checked &&
        (
            targetSizeInput.value.trim() === "" ||
            !Number.isFinite(targetSizeKB) ||
            targetSizeKB <= 0
        )
    ) {
        targetSizeInput.setCustomValidity(
            "Enter a target image size greater than 0 KB."
        );

        targetSizeInput.reportValidity();
        targetSizeInput.focus();
        return false;
    }

    return true;
}


function buildPipelineFormData() {
    const formData = new FormData();

    files.forEach((file) => {
        formData.append("files", file, file.name);
    });

    formData.append("skip_compression", String(!$("#compression").checked));
    formData.append("detect_ai", String($("#ai").checked));
    formData.append("lossless", "false");
    formData.append(
        "max_size_kb",
        $("#compression").checked
            ? $("#target-size-kb").value
            : "500"
    );
    formData.append(
        "retention_rate",
        String(Number($("#retention").value) / 100)
    );

    return formData;
}


let progressTimer = null;
let simulatedProgress = 0;

function setPipelineProgress(percent, title, message, state = "running", errorDetails = "") {
    const container = document.getElementById("pipeline-progress");
    const fill = document.getElementById("progress-fill");
    const percentLabel = document.getElementById("progress-percent");
    const progressBar = document.getElementById("progress-bar");
    const titleLabel = document.getElementById("progress-title");
    const messageLabel = document.getElementById("progress-message");
    const errorBox = document.getElementById("progress-error");

    container.hidden = false;
    container.classList.toggle("is-error", state === "error");
    container.classList.toggle("is-success", state === "success");
    const safePercent = Math.max(0, Math.min(100, Math.round(percent)));
    fill.style.width = `${safePercent}%`;
    percentLabel.textContent = `${safePercent}%`;
    progressBar.setAttribute("aria-valuenow", String(safePercent));
    titleLabel.textContent = title;
    messageLabel.textContent = message;
    errorBox.hidden = !errorDetails;
    errorBox.textContent = errorDetails;
}

function stopSimulatedProgress() {
    if (progressTimer !== null) {
        clearInterval(progressTimer);
        progressTimer = null;
    }
}

function startSimulatedProgress() {
    stopSimulatedProgress();
    simulatedProgress = 3;
    setPipelineProgress(
        simulatedProgress,
        "Processing images…",
        "Uploading images and running the pipeline. Progress is an estimate."
    );

    progressTimer = setInterval(() => {
        // This is deliberately simulated: it advances slowly and never reaches
        // completion until FastAPI confirms that the pipeline succeeded.
        const remaining = 91 - simulatedProgress;
        const increment = Math.max(0.25, Math.min(4, remaining * 0.075));
        simulatedProgress = Math.min(91, simulatedProgress + increment);
        setPipelineProgress(
            simulatedProgress,
            "Processing images…",
            simulatedProgress >= 91
                ? "The pipeline is still running. Waiting for backend confirmation…"
                : "Processing your images. This progress indicator is an estimate."
        );
    }, 650);
}

async function readResponseBody(response) {
    const text = await response.text();
    if (!text) return {};
    try {
        return JSON.parse(text);
    } catch {
        return { detail: text };
    }
}

function describePipelineError(response, responseData) {
    const detail = responseData?.detail ?? responseData?.message ?? responseData?.error;
    let message = "";

    if (typeof detail === "string") {
        message = detail;
    } else if (detail && typeof detail === "object") {
        message = detail.message || detail.error || JSON.stringify(detail, null, 2);
    } else if (detail != null) {
        message = String(detail);
    }

    if (!message) {
        message = response
            ? `The server returned HTTP ${response.status}${response.statusText ? ` (${response.statusText})` : ""}.`
            : "Could not connect to the backend.";
    }

    if (response && !response.ok && !message.includes(`HTTP ${response.status}`)) {
        message = `HTTP ${response.status}: ${message}`;
    }

    return message;
}

async function runPipeline() {
    if (isPipelineRunning || !validatePipelineSettings()) {
        return;
    }

    isPipelineRunning = true;

    const runButton = $("#preview-run");
    const originalButtonText = runButton.textContent;
    let response = null;

    runButton.disabled = true;
    runButton.textContent = "Processing images…";
    startSimulatedProgress();
    toast("Pipeline started. Progress shown here is an estimate.");

    try {
        response = await fetch(
            `${API_BASE_URL}/api/upload-images`,
            {
                method: "POST",
                body: buildPipelineFormData()
            }
        );

        const responseData = await readResponseBody(response);

        if (!response.ok) {
            throw new Error(describePipelineError(response, responseData));
        }

        if (
            responseData.status !== "completed" ||
            !responseData.dashboard
        ) {
            throw new Error(
                describePipelineError(
                    response,
                    { detail: "The backend response did not confirm successful completion or include a dashboard report." }
                )
            );
        }

        // The backend has confirmed success: only now may progress reach 100%.
        stopSimulatedProgress();
        setPipelineProgress(
            100,
            "Pipeline completed successfully",
            "The backend confirmed completion. Loading your results…",
            "success"
        );

        report = responseData.dashboard;
        renderDashboard();
        await loadDatasetGalleries();

        // Navigate only after the backend confirms success and the report is rendered.
        showPage("results");
        toast("Pipeline completed successfully.");

        console.log("Pipeline response:", responseData);
        console.log("Dashboard report:", report);

    } catch (error) {
        stopSimulatedProgress();
        const isConnectionError = error instanceof TypeError;
        const errorMessage = isConnectionError
            ? `Cannot reach FastAPI at ${API_BASE_URL}. Make sure the backend is running and the address/port are correct.`
            : (error.message || "An unexpected error occurred while processing the images.");

        setPipelineProgress(
            simulatedProgress,
            "Processing failed",
            "The pipeline did not complete. Check the error below, fix the issue, and try again.",
            "error",
            errorMessage
        );
        console.error("Pipeline execution failed:", error);
        toast("Pipeline failed. See the error message below the Run pipeline button.");

    } finally {
        stopSimulatedProgress();
        isPipelineRunning = false;
        runButton.disabled = false;
        runButton.textContent = originalButtonText;
    }
}


// ============================================================
// DATASET GALLERIES — ORIGINAL INPUT AND FINAL SELECTED IMAGES
// ============================================================

async function loadGallery(kind, galleryId, countId) {
    const gallery = document.getElementById(galleryId);
    const countLabel = document.getElementById(countId);

    gallery.replaceChildren();
    const loading = document.createElement("p");
    loading.className = "gallery-empty";
    loading.textContent = "Loading images…";
    gallery.appendChild(loading);

    const response = await fetch(
        `${API_BASE_URL}${DATASET_IMAGES_ENDPOINT}/${kind}`
    );

    if (!response.ok) {
        throw new Error(`Could not load ${kind} images (HTTP ${response.status}).`);
    }

    const filenames = await response.json();
    gallery.replaceChildren();

    if (!Array.isArray(filenames) || filenames.length === 0) {
        const empty = document.createElement("p");
        empty.className = "gallery-empty";
        empty.textContent = "No images are available for this run.";
        gallery.appendChild(empty);
        countLabel.textContent = "0 images";
        return;
    }

    countLabel.textContent = `${filenames.length} image(s)`;

    filenames.forEach((filename) => {
        const card = document.createElement("figure");
        card.className = "gallery-image-card";

        const image = document.createElement("img");
        image.loading = "lazy";
        image.alt = filename;
        image.src = `${API_BASE_URL}${DATASET_IMAGES_ENDPOINT}/${kind}/${encodeURIComponent(filename)}`;
        image.addEventListener("error", () => {
            image.alt = `Could not load ${filename}`;
            card.classList.add("image-error");
        }, { once: true });

        const caption = document.createElement("figcaption");
        caption.textContent = filename;
        caption.title = filename;

        card.append(image, caption);
        gallery.appendChild(card);
    });
}

async function loadDatasetGalleries() {
    const downloadButtons = [
        document.getElementById("download-final"),
        document.getElementById("download-final-gallery")
    ];

    downloadButtons.forEach((button) => {
        button.disabled = true;
    });

    try {
        await Promise.all([
            loadGallery("original", "original-gallery", "original-gallery-count"),
            loadGallery("final", "final-gallery", "final-gallery-count")
        ]);

        latestRunHasImages = true;
        downloadButtons.forEach((button) => {
            button.disabled = false;
        });
    } catch (error) {
        latestRunHasImages = false;
        console.error("Could not load dataset galleries:", error);
        toast("Results loaded, but image galleries could not be fetched. Check FastAPI routes.");
    }
}

function downloadFinalDataset() {
    if (!latestRunHasImages) {
        toast("Run the pipeline successfully before downloading the final dataset.");
        return;
    }

    const link = document.createElement("a");
    link.href = `${API_BASE_URL}${FINAL_DATASET_DOWNLOAD_ENDPOINT}`;
    link.download = "final_dataset.zip";
    document.body.appendChild(link);
    link.click();
    link.remove();
}

document.getElementById("download-final").addEventListener("click", downloadFinalDataset);
document.getElementById("download-final-gallery").addEventListener("click", downloadFinalDataset);


// ============================================================
// 8. LOAD AND EXPORT DASHBOARD JSON
// ============================================================

$("#load").addEventListener("click", () => {
    $("#json-file").click();
});


$("#json-file").addEventListener("change", async (event) => {
    const selectedFile = event.target.files[0];

    if (!selectedFile) {
        return;
    }

    try {
        const loadedReport = JSON.parse(await selectedFile.text());

        if (
            !loadedReport ||
            typeof loadedReport !== "object" ||
            !loadedReport.input ||
            !loadedReport.final_dataset
        ) {
            throw new Error("This file is not a valid dashboard report.");
        }

        report = loadedReport;
        renderDashboard();
        latestRunHasImages = false;
        document.getElementById("download-final").disabled = true;
        document.getElementById("download-final-gallery").disabled = true;

        toast("Dashboard JSON loaded. Image galleries require a completed local pipeline run.");

    } catch (error) {
        console.error("Could not load dashboard JSON:", error);

        toast("Could not read that dashboard JSON.");

    } finally {
        event.target.value = "";
    }
});


// ============================================================
// 9. RENDER RESULTS AND ANALYTICS
// ============================================================

function renderDashboard() {
    if (!report) {
        resetDashboard();
        return;
    }

    const input = report.input || {};
    const compression = report.compression || {};
    const quality = report.quality_filtering || {};
    const aiDetection = report.ai_detection || {};
    const preservation = report.information_preservation || {};
    const finalDataset = report.final_dataset || {};

    const originalImageCount = Number(input.image_count) || 0;
    const originalSizeMB = Number(input.total_size_mb) || 0;

    const finalImageCount = Number(finalDataset.image_count) || 0;
    const finalSizeMB = Number(finalDataset.total_size_mb) || 0;

    const sizeSavedMB = originalSizeMB - finalSizeMB;

    const sizeReductionPercent = originalSizeMB
        ? sizeSavedMB / originalSizeMB * 100
        : 0;

    const readableImageCount =
        Number(preservation.readable_image_count) || 0;

    const selectedImageCount =
        preservation.selected_image_count == null
            ? finalImageCount
            : Number(preservation.selected_image_count);

    const actualRetentionPercent =
        preservation.actual_retention_rate_percent == null
            ? (
                readableImageCount
                    ? finalImageCount / readableImageCount * 100
                    : 0
            )
            : Number(preservation.actual_retention_rate_percent);


    // Main metrics.
    $("#orig-count").textContent = originalImageCount;

    $("#orig-size").innerHTML =
        `${formatNumber(originalSizeMB)} <em>MB</em>`;

    $("#final-count").textContent = finalImageCount;

    $("#saved").innerHTML =
        `${formatNumber(sizeSavedMB)} <em>MB</em>`;

    $("#reduction").textContent =
        `${formatNumber(sizeReductionPercent)}% total size reduction`;


    // Compression status.
    $("#comp-status").textContent =
        compression.status || "Unknown";

    $("#comp-reduction").textContent =
        `${formatNumber(compression.size_reduction_percent)}%`;


    // Quality filtering status.
    $("#quality-status").textContent =
        `${quality.status || "Unknown"} · ` +
        `${Number(preservation.input_image_count) || 0} images passed`;

    $("#blurry").textContent =
        Number(quality.blurry_images_removed) || 0;


    // Information preservation status.
    $("#preserve-status").textContent =
        `${formatNumber(
            preservation.configured_retention_rate_percent,
            0
        )}% target retention`;

    $("#actual").textContent =
        `${formatNumber(actualRetentionPercent)}%`;


    // AI detection status.
    $("#ai-status").textContent =
        aiDetection.status === "skipped"
            ? "Skipped · Not run"
            : aiDetection.status || "Unknown";

    $("#ai-count").textContent =
        aiDetection.potential_ai_images == null
            ? "—"
            : aiDetection.potential_ai_images;


    // Retention donut chart.
    $("#donut-value").textContent =
        `${formatNumber(actualRetentionPercent)}%`;

    const safeRetentionPercent = Math.min(
        100,
        Math.max(0, actualRetentionPercent)
    );

    $("#donut").style.background =
        `conic-gradient(
            #5c8fff 0 ${safeRetentionPercent}%,
            #25344b ${safeRetentionPercent}% 100%
        )`;


    // Selected and unselected images.
    $("#selected").textContent = selectedImageCount;

    $("#not-selected").textContent = Math.max(
        0,
        readableImageCount - selectedImageCount
    );


    // Final dataset summary.
    $("#final-summary").textContent =
        `${finalImageCount} images · ` +
        `${formatNumber(finalSizeMB)} MB · ` +
        `Information preservation ${preservation.status || "unknown"}.`;


    // Original versus final dataset bar chart.
    $("#bar-original-label").textContent =
        `${formatNumber(originalSizeMB)} MB`;

    $("#bar-final-label").textContent =
        `${formatNumber(finalSizeMB)} MB`;

    const maximumSize = Math.max(
        originalSizeMB,
        finalSizeMB,
        1
    );

    $("#bar-original").style.height =
        `${originalSizeMB / maximumSize * 100}%`;

    $("#bar-final").style.height =
        `${finalSizeMB / maximumSize * 100}%`;


    // Dataset breakdown.
    const blurryCount =
        Number(quality.blurry_images_removed) || 0;

    const duplicateCount =
        Number(quality.duplicate_images_removed) || 0;

    $("#breakdown-total").textContent = originalImageCount;

    $("#break-final").textContent = finalImageCount;

    $("#break-blurry").textContent = blurryCount;

    $("#break-duplicates").textContent = duplicateCount;

    $("#break-other").textContent = Math.max(
        0,
        originalImageCount -
            finalImageCount -
            blurryCount -
            duplicateCount
    );


    // Dataset breakdown donut chart.
    const finalPercent = originalImageCount
        ? finalImageCount / originalImageCount * 100
        : 0;

    const blurryPercent = originalImageCount
        ? (finalImageCount + blurryCount) /
            originalImageCount * 100
        : 0;

    const duplicatePercent = originalImageCount
        ? (
            finalImageCount +
            blurryCount +
            duplicateCount
        ) / originalImageCount * 100
        : 0;

    $("#breakdown-donut").style.background =
        `conic-gradient(
            #45e2b1 0 ${finalPercent}%,
            #ff718b ${finalPercent}% ${blurryPercent}%,
            #a087ff ${blurryPercent}% ${duplicatePercent}%,
            #34445c ${duplicatePercent}% 100%
        )`;
}


// ============================================================
// 10. EXPORT DASHBOARD REPORT
// ============================================================

$("#export").addEventListener("click", () => {
    if (!report) {
        toast("Run the pipeline or load a dashboard report before exporting.");
        return;
    }

    const reportBlob = new Blob(
        [JSON.stringify(report, null, 2)],
        { type: "application/json" }
    );

    const downloadURL = URL.createObjectURL(reportBlob);
    const downloadLink = document.createElement("a");

    downloadLink.href = downloadURL;
    downloadLink.download = "dashboard_report.json";
    downloadLink.click();

    URL.revokeObjectURL(downloadURL);

    toast("Dashboard JSON exported.");
});


// ============================================================
// 11. INITIAL DASHBOARD STATE
// ============================================================

function resetDashboard() {
    const valueIds = [
        "orig-count",
        "final-count",
        "blurry",
        "ai-count",
        "selected",
        "not-selected",
        "breakdown-total",
        "break-final",
        "break-blurry",
        "break-duplicates",
        "break-other"
    ];

    valueIds.forEach((id) => {
        $(`#${id}`).textContent = "—";
    });

    $("#orig-size").textContent = "—";
    $("#saved").textContent = "—";
    $("#reduction").textContent = "Run the pipeline to see actual results.";
    $("#comp-status").textContent = "Not run";
    $("#comp-reduction").textContent = "—";
    $("#quality-status").textContent = "Not run";
    $("#preserve-status").textContent = "Not run";
    $("#actual").textContent = "—";
    $("#ai-status").textContent = "Not run";
    $("#donut-value").textContent = "—";
    $("#final-summary").textContent = "No pipeline report available yet.";
    $("#bar-original-label").textContent = "—";
    $("#bar-final-label").textContent = "—";
    $("#breakdown-total").textContent = "—";

    $("#donut").style.background =
        "conic-gradient(#25344b 0 100%)";
    $("#breakdown-donut").style.background =
        "conic-gradient(#34445c 0 100%)";
    $("#bar-original").style.height = "0%";
    $("#bar-final").style.height = "0%";
}

resetDashboard();
