(function () {
  "use strict";

  function getPageText(content) {
    const clone = content.cloneNode(true);
    const wrapper = clone.querySelector(".copy-all-wrapper");
    if (wrapper) {
      wrapper.remove();
    }
    return clone.innerText.trim();
  }

  function fallbackCopyText(text) {
    const textarea = document.createElement("textarea");
    textarea.value = text;
    textarea.setAttribute("readonly", "");
    textarea.style.position = "absolute";
    textarea.style.left = "-9999px";
    document.body.appendChild(textarea);
    textarea.select();
    document.execCommand("copy");
    document.body.removeChild(textarea);
  }

  function showCopied(button) {
    const original = button.textContent;
    button.textContent = "Copied!";
    button.classList.add("copy-all-button--done");
    window.setTimeout(function () {
      button.textContent = original;
      button.classList.remove("copy-all-button--done");
    }, 2000);
  }

  function ensureCopyAllButton() {
    const content = document.querySelector("article.md-content__inner");
    if (!content) {
      return;
    }

    if (content.querySelector(".copy-all-button")) {
      return;
    }

    const wrapper = document.createElement("div");
    wrapper.className = "copy-all-wrapper";

    const button = document.createElement("button");
    button.type = "button";
    button.className = "copy-all-button md-button";
    button.setAttribute("aria-label", "Copy entire page");
    button.textContent = "Copy all";

    button.addEventListener("click", function () {
      const text = getPageText(content);
      if (!text) {
        return;
      }

      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard
          .writeText(text)
          .then(function () {
            showCopied(button);
          })
          .catch(function () {
            fallbackCopyText(text);
            showCopied(button);
          });
      } else {
        fallbackCopyText(text);
        showCopied(button);
      }
    });

    wrapper.appendChild(button);
    content.prepend(wrapper);
  }

  if (window.document$ && typeof window.document$.subscribe ==== "function") {
    window.document$.subscribe(ensureCopyAllButton);
  } else {
    window.addEventListener("DOMContentLoaded", ensureCopyAllButton);
  }
})();
