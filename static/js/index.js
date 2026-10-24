// Scroll-to-top button, the banner and teaser videos, and lazy page videos (no jQuery, no carousel).
//
// A visitor who asks for reduced motion (prefers-reduced-motion: reduce) gets no autoplay: the
// banner and the teaser show their posters, and the lazy videos load but wait for a click on play.

var reducedMotion = window.matchMedia
  ? window.matchMedia('(prefers-reduced-motion: reduce)')
  : { matches: false };

function scrollToTop() {
  window.scrollTo({ top: 0, behavior: reducedMotion.matches ? 'auto' : 'smooth' });
}

window.addEventListener('scroll', function () {
  var b = document.querySelector('.scroll-to-top');
  if (!b) return;
  if (window.pageYOffset > 300) b.classList.add('visible');
  else b.classList.remove('visible');
});

document.addEventListener('DOMContentLoaded', function () {
  // Banner and teaser: muted loops that start on their own unless reduced motion is requested.
  var auto = Array.prototype.slice.call(document.querySelectorAll('video[data-autoplay]'));
  function startAuto() {
    if (reducedMotion.matches) return;
    auto.forEach(function (v) {
      if (!v.dataset.userPaused) v.play().catch(function () {});
    });
  }
  startAuto();

  // Videos with data-src load only when they come near the viewport, play (muted, looping) while
  // visible, and pause when scrolled away. Without IntersectionObserver they load right away.
  var vids = Array.prototype.slice.call(document.querySelectorAll('video[data-src]'));
  function load(v) {
    if (v.dataset.loaded) return;
    v.dataset.loaded = '1';
    v.src = v.dataset.src;
    v.preload = reducedMotion.matches ? 'metadata' : 'auto';
  }

  // A visitor who pauses a video keeps it paused when it scrolls back into view.
  auto.concat(vids).forEach(function (v) {
    v.addEventListener('pause', function () {
      if (v.dataset.autoPaused) delete v.dataset.autoPaused;
      else if (document.visibilityState === 'visible') v.dataset.userPaused = '1';
    });
    v.addEventListener('play', function () { delete v.dataset.userPaused; });
  });

  // Switching reduced motion on while the page is open stops the moving videos.
  function onMotionChange() {
    if (!reducedMotion.matches) {
      startAuto();
      return;
    }
    auto.concat(vids).forEach(function (v) {
      if (!v.paused) {
        v.dataset.autoPaused = '1';
        v.pause();
      }
    });
  }
  if (reducedMotion.addEventListener) reducedMotion.addEventListener('change', onMotionChange);
  else if (reducedMotion.addListener) reducedMotion.addListener(onMotionChange);

  if (!('IntersectionObserver' in window)) {
    vids.forEach(load);
    return;
  }
  var near = new IntersectionObserver(function (entries) {
    entries.forEach(function (e) {
      if (e.isIntersecting) {
        load(e.target);
        near.unobserve(e.target);
      }
    });
  }, { rootMargin: '300px 0px' });
  var seen = new IntersectionObserver(function (entries) {
    entries.forEach(function (e) {
      var v = e.target;
      if (e.isIntersecting) {
        load(v);
        if (!v.dataset.userPaused && !reducedMotion.matches) v.play().catch(function () {});
      } else if (!v.paused) {
        v.dataset.autoPaused = '1';
        v.pause();
      }
    });
  }, { threshold: 0.35 });
  vids.forEach(function (v) {
    near.observe(v);
    seen.observe(v);
  });
});

// Copy the BibTeX block to the clipboard (button in the BibTeX section).
function copyBibTeX() {
  const code = document.getElementById('bibtex-code');
  const button = document.querySelector('.copy-bibtex-btn');
  if (!code || !button) return;
  const label = button.querySelector('.copy-text');
  const done = () => {
    button.classList.add('copied');
    label.textContent = 'Cop';
    setTimeout(() => { button.classList.remove('copied'); label.textContent = 'Copy'; }, 2000);
  };
  if (navigator.clipboard && navigator.clipboard.writeText) {
    navigator.clipboard.writeText(code.textContent).then(done).catch(() => {});
  }
}

// Film player: any [data-film] link opens the film full screen and plays it; closing (button, Esc,
// a click outside the video, or the end) unloads it. Without <dialog> support the link opens the file.
(function () {
  const film = document.getElementById('film');
  const video = document.getElementById('filmV');
  const close = document.getElementById('filmX');
  if (!film || !video || typeof film.showModal !== 'function') return;
  const shut = () => { if (film.open) film.close(); };
  video.addEventListener('ended', shut);
  close.addEventListener('click', shut);
  film.addEventListener('click', (e) => { if (e.target === film) shut(); });
  film.addEventListener('close', () => {
    video.pause(); video.removeAttribute('src'); video.load();
    document.documentElement.style.overflow = '';
  });
  document.querySelectorAll('[data-film]').forEach((a) => {
    a.addEventListener('click', (e) => {
      e.preventDefault();
      film.showModal();
      document.documentElement.style.overflow = 'hidden';
      video.setAttribute('src', video.dataset.src);
      const p = video.play();
      if (p && p.catch) p.catch(() => {});
      close.focus();
    });
  });
})();
