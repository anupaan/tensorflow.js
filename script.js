let model;
const MODEL_SAVE_PATH = 'indexeddb://my-mnist-model';
 
// Load MNIST Data from local JSON
async function loadMnistData(isTrain = true) {
    console.log("Loading MNIST data...");
    const filePath = isTrain ? 'mnist_handwritten_train.json' : 'mnist_handwritten_test.json';
 
    try {
        const response = await fetch(filePath);
        const text = await response.text();
        const lines = text.split("\n").filter(line => line.trim() !== "");
        const images = [];
        const labels = [];
 
        for (const line of lines) {
            try {
                const data = JSON.parse(line);
                const normalized = data.image.map(p => p / 255);

                images.push(normalized);
                labels.push(data.label);
            } catch (error) {
                console.error("JSON parse error: ", error);
            }
        }
 
        return { images, labels };
    } catch (error) {
        console.error("Error loading data:", error);
        document.getElementById('status').innerText = 'Status: Error loading data';
        return {};
    }
}
 
// Create model
async function createModel() {
    console.log("Creating model...");
    model = tf.sequential();
    model.add(tf.layers.dense({ inputShape: [784], units: 128, activation: 'relu' }));
    model.add(tf.layers.dense({ units: 10, activation: 'softmax' }));
    model.compile({
        optimizer: 'adam',
        loss: 'categoricalCrossentropy',
        metrics: ['accuracy']
    });
    console.log("✅ Model created!");
    return model;
}
 
// Train and Save
async function trainModel() {
    document.getElementById('status').innerText = 'Status: Training...';
    try {
        model = await createModel();
        const { images, labels } = await loadMnistData(true);
        const xTrain = tf.tensor2d(images, [images.length, 784]);
        const yTrain = tf.oneHot(tf.tensor1d(labels, 'int32'), 10);
 
        await model.fit(xTrain, yTrain, {
            epochs: 10,
            batchSize: 32,
            validationSplit: 0.1,
            callbacks: {
                onEpochEnd: (epoch, logs) => {
                    console.log(`Epoch ${epoch + 1}: loss = ${logs.loss}, acc = ${logs.acc}, val_loss = ${logs.val_loss}`);
                }
            }
        });
 
        await model.save(MODEL_SAVE_PATH);
        document.getElementById('status').innerText = 'Status: Training Complete (Model Saved)';
    } catch (error) {
        console.error("Training error:", error);
        document.getElementById('status').innerText = 'Status: Training Error';
    }
}
 
// Load model
async function loadModel() {
    try {
        model = await tf.loadLayersModel(MODEL_SAVE_PATH);
        console.log("✅ Model loaded from IndexedDB");
        document.getElementById('status').innerText = 'Status: Model Loaded';
    } catch (error) {
        console.error("Load error:", error);
        document.getElementById('status').innerText = 'Status: No Saved Model Found';
    }
}
 
// Predict on test data
async function testModel() {
    if (!model) await loadModel();
    if (!model) {
        alert("Train the model first!");
        return;
    }
 
    try {
        const { images, labels } = await loadMnistData(false);
        const index = Math.floor(Math.random() * images.length);
        const sample = tf.tensor2d([images[index]], [1, 784]);
        const prediction = model.predict(sample);
        const predictedClass = prediction.argMax(1).dataSync()[0];
 
        document.getElementById('status').innerText = `Predicted Digit: ${predictedClass} (Actual: ${labels[index]})`;
        drawMnistImage(sample, "mnistCanvas");
    } catch (error) {
        console.error("Prediction error: ", error);
        document.getElementById('status').innerText = 'Status: Prediction Error';
    }
}
 
// Draw MNIST Image on Canvas
function drawMnistImage(pixelTensor, canvasId) {
    const canvas = document.getElementById(canvasId);
    const ctx = canvas.getContext('2d');
    const pixelData = pixelTensor.dataSync();
    const imageData = ctx.createImageData(28, 28);
 
    for (let i = 0; i < 784; i++) {
        const val = pixelData[i] * 255;
        const j = i * 4;
        imageData.data[j] = val;
        imageData.data[j + 1] = val;
        imageData.data[j + 2] = val;
        imageData.data[j + 3] = 255;
    }
 
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.putImageData(imageData, 0, 0);
}
 
// OPTIONAL: Live Draw + Predict (if using drawCanvas)
function predictFromDrawing() {
    if (!model) {
        alert("Train or load model first! ");
        return;
    }
 
    const canvas = document.getElementById('drawCanvas');
    const ctx = canvas.getContext('2d');
    const imageData = ctx.getImageData(0, 0, 280, 280);
    const pixels = new Float32Array(784);
 
    // Downscale 280x280 to 28x28 manually
    const scale = 10;
    for (let y = 0; y < 28; y++) {
        for (let x = 0; x < 28; x++) {
            let sum = 0;
            for (let dy = 0; dy < scale; dy++) {
                for (let dx = 0; dx < scale; dx++) {
                    const px = (y * scale + dy) * 280 + (x * scale + dx);
                    sum += imageData.data[px * 4]; // red channel
                }
            }
            pixels[y * 28 + x] = (sum / (scale * scale * 255));
        }
    }
 
    const input = tf.tensor2d([pixels], [1, 784]);
    const prediction = model.predict(input);
    const predicted = prediction.argMax(1).dataSync()[0];
 
    document.getElementById('status').innerText = `Your drawn digit is: ${predicted}`;
    drawMnistImage(input, "mnistCanvas");
}