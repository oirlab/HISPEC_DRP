(function () {
  function saveTextFile(filename, text) {
    var blob = new Blob([text], { type: "application/octet-stream" });
    var url = URL.createObjectURL(blob);
    var a = document.createElement("a");
    a.href = url;
    a.download = filename;
    a.rel = "noopener";
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(function () {
      URL.revokeObjectURL(url);
    }, 0);
  }

  document.addEventListener(
    "click",
    function (event) {
      var btn = event.target.closest("a[data-notebook-download]");
      if (!btn) {
        return;
      }
      event.preventDefault();
      event.stopPropagation();

      var filename = btn.getAttribute("data-notebook-download");
      var notebooks = window.HISPEC_DRP_NOTEBOOKS || {};
      var text = notebooks[filename];
      if (typeof text === "string" && text.length) {
        saveTextFile(filename, text);
        return;
      }

      var href = btn.getAttribute("href");
      if (!href || href === "#") {
        return;
      }
      fetch(href)
        .then(function (response) {
          if (!response.ok) {
            throw new Error(String(response.status));
          }
          return response.text();
        })
        .then(function (body) {
          saveTextFile(filename, body);
        })
        .catch(function () {
          window.location.href = href;
        });
    },
    true
  );
})();
