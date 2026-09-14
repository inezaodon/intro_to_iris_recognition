(function () {
  const tabs = Array.from(document.querySelectorAll(".tab"));
  const panels = Array.from(document.querySelectorAll(".panel"));
  const slides = Array.from(document.querySelectorAll(".slide"));
  const dotsWrap = document.getElementById("dots");
  const prev = document.getElementById("prev");
  const next = document.getElementById("next");
  const pos = document.getElementById("pos");

  let i = 0;
  let applyingHash = false;

  function selectedTab() {
    return tabs.find((t) => t.getAttribute("aria-selected") === "true")?.dataset.tab || "home";
  }

  function hashFor(tab, slideIndex) {
    return tab === "daugman" ? "#daugman/" + (slideIndex + 1) : "#" + tab;
  }

  function writeHash(tab, slideIndex) {
    const nextHash = hashFor(tab, slideIndex);
    if (location.hash === nextHash) return;
    applyingHash = true;
    location.hash = nextHash;
  }

  function showTab(id, push) {
    tabs.forEach((t) => t.setAttribute("aria-selected", String(t.dataset.tab === id)));
    panels.forEach((p) => p.classList.toggle("active", p.id === id));
    if (push) writeHash(id, i);
  }

  function renderDots() {
    if (!dotsWrap) return;
    dotsWrap.innerHTML = slides
      .map(function (_, n) {
        return (
          '<button type="button" class="dot' +
          (n === i ? " on" : "") +
          '" data-i="' +
          n +
          '" aria-label="Slide ' +
          (n + 1) +
          '"></button>'
        );
      })
      .join("");
  }

  function showSlide(n, push) {
    if (!slides.length) return;
    i = Math.max(0, Math.min(slides.length - 1, n));
    slides.forEach((s, k) => s.classList.toggle("on", k === i));
    renderDots();
    if (pos) pos.textContent = i + 1 + " / " + slides.length;
    if (prev) prev.disabled = i === 0;
    if (next) next.disabled = i === slides.length - 1;
    if (push && selectedTab() === "daugman") writeHash("daugman", i);
  }

  tabs.forEach((t) => {
    t.addEventListener("click", (e) => {
      e.preventDefault();
      showTab(t.dataset.tab, true);
    });
  });

  prev?.addEventListener("click", (e) => {
    e.preventDefault();
    showSlide(i - 1, true);
  });
  next?.addEventListener("click", (e) => {
    e.preventDefault();
    showSlide(i + 1, true);
  });
  dotsWrap?.addEventListener("click", (e) => {
    const d = e.target.closest(".dot");
    if (!d) return;
    e.preventDefault();
    showSlide(Number(d.dataset.i), true);
  });

  document.addEventListener("keydown", (e) => {
    if (selectedTab() !== "daugman") return;
    if (["INPUT", "TEXTAREA"].includes(document.activeElement?.tagName)) return;
    if (["ArrowRight", "PageDown", " ", "l"].includes(e.key)) {
      e.preventDefault();
      showSlide(i + 1, true);
    } else if (["ArrowLeft", "PageUp", "h"].includes(e.key)) {
      e.preventDefault();
      showSlide(i - 1, true);
    } else if (e.key === "Home") {
      e.preventDefault();
      showSlide(0, true);
    } else if (e.key === "End") {
      e.preventDefault();
      showSlide(slides.length - 1, true);
    }
  });

  document.querySelectorAll(".quiz .q").forEach((btn) => {
    btn.addEventListener("click", () => btn.classList.toggle("open"));
  });

  function bootFromHash() {
    if (applyingHash) {
      applyingHash = false;
      return;
    }
    const raw = (location.hash || "").replace("#", "");
    const parts = raw.split("/");
    const tab = parts[0];
    const slide = parts[1];
    const id = tabs.some((t) => t.dataset.tab === tab) ? tab : "home";
    showTab(id, false);
    const n = slide ? Number(slide) - 1 : 0;
    showSlide(Number.isFinite(n) ? n : 0, false);
  }

  window.addEventListener("hashchange", bootFromHash);
  bootFromHash();
})();
