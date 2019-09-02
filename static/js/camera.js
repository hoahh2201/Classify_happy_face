// Prefer camera resolution nearest to 1280x720.
const constraints = { audio: false, video: { width: 1280, height: 720 } };

navigator.mediaDevices.getUserMedia(constraints)
.then(function(mediaStream) {
  let video = document.querySelector('video'),
      canvas = document.getElementById('canvas'),
      context = canvas.getContext('2d');
  video.srcObject = mediaStream;
  video.onloadedmetadata = function(e) {
      video.play().then(() => {
          // adding your optional function
      }).catch(() => {
            // Auto-play was prevented
            // Show paused UI.
      });
document.getElementById('capture').addEventListener('click', function (me) {
    context.drawImage(video, 0, 0, 400, 300)
});
  };
})
.catch(function(err) {
    console.log(err.name + ": " + err.message);
        });
 // always check for errors at the end.