(() => {
  const requestFullscreen = (video) => {
    if (document.fullscreenElement === video) {
      return;
    }

    if (video.requestFullscreen) {
      void video.requestFullscreen();
      return;
    }

    if (video.webkitEnterFullscreen) {
      video.webkitEnterFullscreen();
      return;
    }

    if (video.webkitRequestFullscreen) {
      video.webkitRequestFullscreen();
    }
  };

  const bindShowcaseVideos = () => {
    document.querySelectorAll("video[data-showcase-video]").forEach((video) => {
      if (video.dataset.fullscreenBound === "true") {
        return;
      }

      video.dataset.fullscreenBound = "true";
      video.addEventListener("click", () => requestFullscreen(video));
      video.addEventListener("keydown", (event) => {
        if (event.key !== "Enter" && event.key !== " ") {
          return;
        }

        event.preventDefault();
        requestFullscreen(video);
      });
    });
  };

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", bindShowcaseVideos);
  } else {
    bindShowcaseVideos();
  }

  if (window.document$) {
    window.document$.subscribe(bindShowcaseVideos);
  }
})();
