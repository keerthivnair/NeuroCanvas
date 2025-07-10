var ocrDemo = {
  CANVAS_WIDTH: 200,
  TRANSLATED_WIDTH: 20,
  PIXEL_WIDTH: 10,
  BLUE: "#0000FF",
  data: Array(400).fill(0),

  // Neural network config
  BATCH_SIZE: 1, // Send after 5 training samples
  trainingRequestCount: 0, // Counter to track how many have been collected
  trainArray: [],

  //Server configuartion
  HOST: "http://localhost", // Server URL
  PORT: "8000",

  // ctx - canvas rendering context(to get the context out of the canvas)
  drawGrid: function (ctx) {
    for (
      var x = this.PIXEL_WIDTH, y = this.PIXEL_WIDTH;
      x < this.CANVAS_WIDTH;
      x += this.PIXEL_WIDTH, y += this.PIXEL_WIDTH
    ) {
      ctx.strokeStyle = this.BLUE;
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, this.CANVAS_WIDTH);
      ctx.stroke();

      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(this.CANVAS_WIDTH, y);
      ctx.stroke();
    }
  },

  onMouseMove: function (e, ctx, canvas) {
    if (!canvas.isDrawing) return;
    this.fillSquare(
      ctx,
      e.clientX - canvas.offsetLeft,
      e.clientY - canvas.offsetTop
    );
  },
  onMouseDown: function (e, ctx, canvas) {
    canvas.isDrawing = true;
    this.fillSquare(
      ctx,
      e.clientX - canvas.offsetLeft,
      e.clientY - canvas.offsetTop
    );
  },

  onMouseUp: function (e, canvas) {
    canvas.isDrawing = false;
  },

  fillSquare: function (ctx, x, y) {
    var xPixel = Math.floor(x / this.PIXEL_WIDTH);
    var yPixel = Math.floor(y / this.PIXEL_WIDTH);
    this.data[xPixel * this.TRANSLATED_WIDTH + yPixel];

    ctx.fillStyle = "#000000";
    ctx.fillRect(
      xPixel * this.PIXEL_WIDTH,
      yPixel * this.PIXEL_WIDTH,
      this.PIXEL_WIDTH,
      this.PIXEL_WIDTH
    );
  },

  train: function () {
    var digitVal = document.getElementById("digit").value;
    if (!digitVal || this.data.indexOf(1) < 0) {
      alert("Please type and draw a digit value in order to train the network");
      return;
    }

    this.trainArray.push({ y0: this.data, label: parseInt(digitVal) });
    this.trainingRequestCount++;

    if (this.trainingRequestCount == this.BATCH_SIZE) {
      alert("Sending training data to the server...");
      var json = {
        trainArray: this.trainArray,
        train: true,
      };

      this.sendData(json);
      this.trainingRequestCount = 0;
      this.trainArray = [];
    }
    this.resetCanvas()
  },

  test: function () {
    if (this.data.indexOf(1) < 0) {
      alert("Please draw a digit in order to test the network");
      return;
    }
    var json = {
      image: this.data,
      predict: true,
    };
    this.sendData(json);
  },

  resetCanvas: function () {
    var canvas = document.getElementById('canvas');
    var ctx = canvas.getContext('2d');

    ctx.clearRect(0, 0, this.CANVAS_WIDTH, this.CANVAS_WIDTH);
    this.drawGrid(ctx);
    this.data = Array(400).fill(0);
  },

  receiveResponse: function (xmlHttp) {
    if (xmlHttp.status != 200) {
      alert("Server returned status " + xmlHttp.status);
      return;
    }
    var responseJSON = JSON.parse(xmlHttp.responseText);
    if (xmlHttp.responseText && responseJSON.type == "test") {
      alert(
        "The neural network predicts you wrote a '" + responseJSON.result + "'"
      );
    }
  },
  onError: function (e) {
    alert("Error occurred while connecting to server: " + e.target.statusText);
  },

  sendData: function (json) {
    var xmlHttp = new XMLHttpRequest();
    xmlHttp.open("POST", this.HOST + ":" + this.PORT, false);
    xmlHttp.onload = function () {
      this.receiveResponse(xmlHttp);
    }.bind(this);
    xmlHttp.onerror = function () {
      this.onError(xmlHttp);
    }.bind(this);
    var msg = JSON.stringify(json);
    xmlHttp.setRequestHeader("Content-length", msg.length);
    xmlHttp.setRequestHeader("Connection", "close");
    xmlHttp.send(msg);
  },

  onLoadFunction: function () {
    var canvas = document.getElementById("canvas");
    var ctx = canvas.getContext("2d");

    this.drawGrid(ctx);

    canvas.addEventListener('mousedown', (e) => this.onMouseDown(e, ctx, canvas));
    canvas.addEventListener('mousemove', (e) => this.onMouseMove(e, ctx, canvas));
    canvas.addEventListener('mouseup', (e) => this.onMouseUp(e, canvas));
  },
};
