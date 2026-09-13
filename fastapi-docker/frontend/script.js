const imageInput = document.getElementById("imageInput");
const uploadArea = document.getElementById("uploadArea");

const preview = document.getElementById("preview");
const previewContainer = document.getElementById("preview-container");

const predictBtn = document.getElementById("predictBtn");
const btnText = document.getElementById("btnText");
const statusText = document.getElementById("status");

const removeBtn = document.getElementById("removeBtn");
const newAnalysisBtn = document.getElementById("newAnalysisBtn");

const resultSection = document.getElementById("result-section");
const gradcamSection = document.getElementById("gradcam-section");
const newAnalysisContainer = document.getElementById(
    "new-analysis-container"
);

const prediction = document.getElementById("prediction");
const confidence = document.getElementById("confidence");

const covidValue = document.getElementById("covid-value");
const normalValue = document.getElementById("normal-value");
const pneumoniaValue = document.getElementById("pneumonia-value");

const covidBar = document.getElementById("covid-bar");
const normalBar = document.getElementById("normal-bar");
const pneumoniaBar = document.getElementById("pneumonia-bar");

const originalImage = document.getElementById("original-image");
const gradcamImage = document.getElementById("gradcam-image");

let selectedFile = null;
let previewUrl = null;


// =========================
// IMAGE SELECTION
// =========================

imageInput.addEventListener("change", () => {

    const file = imageInput.files[0];

    if (file) {
        handleFile(file);
    }
});


// =========================
// DRAG & DROP
// =========================

uploadArea.addEventListener("dragover", (event) => {

    event.preventDefault();

    uploadArea.classList.add("dragover");
});


uploadArea.addEventListener("dragleave", () => {

    uploadArea.classList.remove("dragover");
});


uploadArea.addEventListener("drop", (event) => {

    event.preventDefault();

    uploadArea.classList.remove("dragover");

    const file = event.dataTransfer.files[0];

    if (file) {
        handleFile(file);
    }
});


// =========================
// HANDLE FILE
// =========================

function handleFile(file) {

    if (!file.type.startsWith("image/")) {

        statusText.textContent =
            "Please select a valid image file.";

        return;
    }

    selectedFile = file;

    // Remove previous object URL
    if (previewUrl) {
        URL.revokeObjectURL(previewUrl);
    }

    previewUrl = URL.createObjectURL(file);

    preview.src = previewUrl;

    previewContainer.style.display = "block";

    predictBtn.disabled = false;

    resultSection.style.display = "none";
    gradcamSection.style.display = "none";
    newAnalysisContainer.style.display = "none";

    statusText.textContent =
        "X-ray selected. Ready for analysis.";

    predictBtn.scrollIntoView({
        behavior: "smooth",
        block: "center"
    });
}


// =========================
// REMOVE IMAGE
// =========================

removeBtn.addEventListener("click", resetPage);

newAnalysisBtn.addEventListener("click", resetPage);


function resetPage() {

    if (previewUrl) {
        URL.revokeObjectURL(previewUrl);
        previewUrl = null;
    }

    selectedFile = null;

    imageInput.value = "";

    preview.src = "";

    originalImage.src = "";
    gradcamImage.src = "";

    previewContainer.style.display = "none";

    resultSection.style.display = "none";
    gradcamSection.style.display = "none";
    newAnalysisContainer.style.display = "none";

    predictBtn.disabled = true;

    btnText.textContent = "Analyze X-Ray";

    predictBtn.classList.remove("loading");

    statusText.textContent = "";

    window.scrollTo({
        top: 0,
        behavior: "smooth"
    });
}


// =========================
// ANALYZE
// =========================

predictBtn.addEventListener("click", async () => {

    if (!selectedFile) {

        statusText.textContent =
            "Please select an X-ray image first.";

        return;
    }


    const formData = new FormData();

    formData.append("image", selectedFile);


    // Loading state
    predictBtn.disabled = true;
    predictBtn.classList.add("loading");

    btnText.textContent = "Analyzing X-Ray...";

    statusText.textContent =
        "AI is analyzing the X-ray. Please wait.";

    resultSection.style.display = "none";
    gradcamSection.style.display = "none";


    try {

        const response = await fetch("/gradcam", {
            method: "POST",
            body: formData
        });


        const data = await response.json();


        if (!response.ok) {

            throw new Error(
                data.detail || "Analysis failed."
            );
        }

        if (!data.is_xray) {
            prediction.textContent = "Not an X-Ray";
            confidence.textContent = `${(data.xray_confidence * 100).toFixed(2)}%`;
            resultSection.style.display = "block";
            gradcamSection.style.display = "none";
            newAnalysisContainer.style.display = "block";
            statusText.textContent = data.message || "The uploaded image was not accepted as a chest X-ray.";
            predictBtn.classList.remove("loading");
            btnText.textContent = "Analysis Complete";
            return;
        }


        // =========================
        // PREDICTION
        // =========================

        prediction.textContent =
            data.predicted_class;


        const confidencePercent =
            data.confidence * 100;


        confidence.textContent =
            `${confidencePercent.toFixed(2)}%`;


        // =========================
        // PROBABILITIES
        // =========================

        const covid =
            data.probabilities.COVID * 100;

        const normal =
            data.probabilities.NORMAL * 100;

        const pneumonia =
            data.probabilities.PNEUMONIA * 100;


        covidValue.textContent =
            `${covid.toFixed(2)}%`;

        normalValue.textContent =
            `${normal.toFixed(2)}%`;

        pneumoniaValue.textContent =
            `${pneumonia.toFixed(2)}%`;


        // Animate bars
        setTimeout(() => {

            covidBar.style.width =
                `${covid}%`;

            normalBar.style.width =
                `${normal}%`;

            pneumoniaBar.style.width =
                `${pneumonia}%`;

        }, 100);


        // =========================
        // ORIGINAL IMAGE
        // =========================

        originalImage.src = preview.src;


        // =========================
        // GRAD-CAM
        // =========================

        gradcamImage.src =
            data.gradcam_image_url;


        // =========================
        // SHOW RESULTS
        // =========================

        resultSection.style.display = "block";

        gradcamSection.style.display = "block";

        newAnalysisContainer.style.display = "block";


        statusText.textContent =
            `X-ray accepted (${(data.xray_confidence * 100).toFixed(2)}% confidence). Analysis completed successfully.`;


        // Reset button appearance
        predictBtn.classList.remove("loading");

        btnText.textContent =
            "Analysis Complete";


        // Scroll
        setTimeout(() => {

            resultSection.scrollIntoView({
                behavior: "smooth",
                block: "start"
            });

        }, 150);


    }

    catch (error) {

        console.error(error);

        statusText.textContent =
            `Unable to analyze image: ${error.message}`;

        resultSection.style.display = "none";
        gradcamSection.style.display = "none";

        predictBtn.classList.remove("loading");

        btnText.textContent =
            "Try Again";

    }

    finally {

        predictBtn.disabled = false;

    }

});
