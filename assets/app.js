const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];

const tabs = $$(".tab");
const panels = $$(".panel");
const slides = $$(".slide");
const dotsWrap = $(".dots");
const prev = $(".prev");
const next = $(".next");
const pos = $(".pos");
let i = 0;

function showTab(id) {
  tabs.forEach((t) => t.setAttribute("aria-selected", String(t.dataset.tab === id)));
  panels.forEach((p) => p.classList.toggle("active", p.id === id));
  location.hash = id === "daugman" ? `#daugman/${i + 1}` : `#${id}`;
}

function renderDots() {
  dotsWrap.innerHTML = slides
    .map((_, n) => `<button class="dot${n === i ? " on" : ""}" data-i="${n}" aria-label="Slide ${n + 1}"></button>`)
    .join("");
}

function showSlide(n) {
  i = Math.max(0, Math.min(slides.length - 1, n));
  slides.forEach((s, k) => s.classList.toggle("on", k === i));
  renderDots();
  pos.textContent = `${i + 1} / ${slides.length}`;
  prev.disabled = i === 0;
  next.disabled = i === slides.length - 1;
  if ($(".tab[aria-selected='true']")?.dataset.tab === "daugman") {
    location.hash = `#daugman/${i + 1}`;
  }
}

tabs.forEach((t) => t.addEventListener("click", () => showTab(t.dataset.tab)));
prev.addEventListener("click", () => showSlide(i - 1));
next.addEventListener("click", () => showSlide(i + 1));
dotsWrap.addEventListener("click", (e) => {
  const d = e.target.closest(".dot");
  if (d) showSlide(Number(d.dataset.i));
});

document.addEventListener("keydown", (e) => {
  const onDeck = $("#daugman").classList.contains("active");
  if (!onDeck) return;
  if (["ArrowRight", "PageDown", " ", "l"].includes(e.key)) {
    e.preventDefault();
    showSlide(i + 1);
  }
  if (["ArrowLeft", "PageUp", "h"].includes(e.key)) {
    e.preventDefault();
    showSlide(i - 1);
  }
  if (e.key === "Home") showSlide(0);
  if (e.key === "End") showSlide(slides.length - 1);
});

$$(".quiz .q").forEach((btn) => {
  btn.addEventListener("click", () => btn.classList.toggle("open"));
});

function boot() {
  const hash = location.hash.replace("#", "");
  const [tab, slide] = hash.split("/");
  const known = tabs.some((t) => t.dataset.tab === tab);
  showTab(known ? tab : "home");
  showSlide(slide ? Number(slide) - 1 : 0);
}

window.addEventListener("hashchange", boot);
boot();
