document$.subscribe(function () {
  if (window.mermaid) {
    mermaid.initialize({
      startOnLoad: true,
      theme: "default"
    });
    mermaid.init(undefined, document.querySelectorAll('.mermaid'));
  }
});