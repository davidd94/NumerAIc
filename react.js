if (canvasRef.current) {
    // this will get the canvas HTML element where everyhting that's painted is drawn
    // and call the toDataURL() function on it
    console.log(canvasRef.current.canvasContainer.children[1].toDataURL());
  }