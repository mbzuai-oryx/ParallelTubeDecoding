document.documentElement.classList.add('motion-ready');

const body = document.body;
const nav = document.querySelector('[data-nav]');

const setActiveSideLink = (hash) => {
  nav?.querySelectorAll('a[href^="#"]').forEach((link) => {
    link.classList.toggle('is-active', link.getAttribute('href') === hash);
  });
};

nav?.querySelectorAll('a[href^="#"]').forEach((link) => {
  link.addEventListener('click', () => setActiveSideLink(link.getAttribute('href')));
});

const revealItems = document.querySelectorAll('.reveal');
if ('IntersectionObserver' in window && !window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
  const revealObserver = new IntersectionObserver((entries, observer) => {
    entries.forEach((entry) => {
      if (!entry.isIntersecting) return;
      entry.target.classList.add('is-visible');
      observer.unobserve(entry.target);
    });
  }, { threshold: 0.11, rootMargin: '0px 0px -45px' });

  revealItems.forEach((item) => revealObserver.observe(item));
} else {
  revealItems.forEach((item) => item.classList.add('is-visible'));
}

const navSections = [...document.querySelectorAll('main section[id]')]
  .filter((section) => document.querySelector(`.side-nav a[href="#${section.id}"]`));

let navSyncQueued = false;

const syncSideNavigation = () => {
  navSyncQueued = false;
  if (!navSections.length) return;

  const atPageEnd = window.scrollY + window.innerHeight >= document.documentElement.scrollHeight - 4;
  if (atPageEnd) {
    setActiveSideLink(`#${navSections[navSections.length - 1].id}`);
    return;
  }

  // Use a point just above the viewport's middle so the active item follows
  // the section the reader is currently viewing in either scroll direction.
  const readingLine = window.scrollY + window.innerHeight * 0.38;
  let activeSection = navSections[0];

  navSections.forEach((section) => {
    if (section.offsetTop <= readingLine) activeSection = section;
  });

  setActiveSideLink(`#${activeSection.id}`);
};

const requestSideNavigationSync = () => {
  if (navSyncQueued) return;
  navSyncQueued = true;
  window.requestAnimationFrame(syncSideNavigation);
};

window.addEventListener('scroll', requestSideNavigationSync, { passive: true });
window.addEventListener('resize', requestSideNavigationSync, { passive: true });
window.addEventListener('load', syncSideNavigation);
syncSideNavigation();

const teaserVideo = document.querySelector('.hero-video-shell video');
teaserVideo?.addEventListener('mouseenter', () => teaserVideo.play().catch(() => {}));

const examplesVideo = document.querySelector('.ptd-examples-video-shell video');
examplesVideo?.addEventListener('mouseenter', () => examplesVideo.play().catch(() => {}));

const speedCarousel = document.querySelector('[data-speed-carousel]');
const speedTrack = speedCarousel?.querySelector('[data-speed-track]');
const speedSlides = [...(speedCarousel?.querySelectorAll('[data-speed-slide]') || [])];
const speedStatus = speedCarousel?.querySelector('[data-speed-status]');
const speedPrevious = speedCarousel?.querySelector('[data-speed-prev]');
const speedNext = speedCarousel?.querySelector('[data-speed-next]');

if (speedCarousel && speedTrack && speedSlides.length) {
  let speedIndex = 0;

  const playSpeedVideo = (video) => {
    video?.play().catch(() => {});
  };

  const showSpeedSlide = (nextIndex, autoplay = false) => {
    speedIndex = (nextIndex + speedSlides.length) % speedSlides.length;
    speedTrack.style.setProperty('--speed-offset', `${speedIndex * -100}%`);

    speedSlides.forEach((slide, index) => {
      const active = index === speedIndex;
      const video = slide.querySelector('video');
      slide.setAttribute('aria-hidden', String(!active));
      if (video) {
        video.tabIndex = active ? 0 : -1;
        if (!active) video.pause();
        if (active && autoplay) playSpeedVideo(video);
      }
    });

    const name = speedSlides[speedIndex].dataset.speedName || `Video ${speedIndex + 1}`;
    if (speedStatus) speedStatus.textContent = `${name} · ${speedIndex + 1} / ${speedSlides.length}`;
  };

  speedSlides.forEach((slide, index) => {
    const video = slide.querySelector('video');
    video?.addEventListener('mouseenter', () => {
      if (index === speedIndex) playSpeedVideo(video);
    });
  });

  speedPrevious?.addEventListener('click', () => showSpeedSlide(speedIndex - 1, true));
  speedNext?.addEventListener('click', () => showSpeedSlide(speedIndex + 1, true));
  speedCarousel.addEventListener('keydown', (event) => {
    if (event.target !== speedCarousel) return;
    if (event.key === 'ArrowLeft') showSpeedSlide(speedIndex - 1, true);
    if (event.key === 'ArrowRight') showSpeedSlide(speedIndex + 1, true);
  });

  showSpeedSlide(0);
}

const lightbox = document.querySelector('[data-lightbox-dialog]');
const lightboxImage = lightbox?.querySelector('img');
const lightboxCaption = lightbox?.querySelector('p');
const lightboxClose = document.querySelector('[data-lightbox-close]');

document.querySelectorAll('[data-lightbox]').forEach((trigger) => {
  trigger.addEventListener('click', () => {
    if (!lightbox || !lightboxImage || !lightboxCaption) return;
    lightboxImage.src = trigger.dataset.lightbox;
    lightboxImage.alt = trigger.querySelector('img')?.alt || 'Expanded research figure';
    lightboxCaption.textContent = trigger.dataset.caption || '';
    body.classList.add('is-lightbox-open');
    lightbox.showModal();
  });
});

const closeLightbox = () => {
  if (!lightbox?.open) return;
  lightbox.close();
  body.classList.remove('is-lightbox-open');
};

lightboxClose?.addEventListener('click', closeLightbox);
lightbox?.addEventListener('click', (event) => {
  if (event.target === lightbox) closeLightbox();
});
lightbox?.addEventListener('close', () => body.classList.remove('is-lightbox-open'));

const copyButton = document.querySelector('[data-copy-bib]');
const bibtex = document.querySelector('[data-bibtex]');

copyButton?.addEventListener('click', async () => {
  if (!bibtex) return;
  const label = copyButton.querySelector('span');
  try {
    await navigator.clipboard.writeText(bibtex.textContent.trim());
    if (label) label.textContent = 'Copied';
  } catch {
    const selection = window.getSelection();
    const range = document.createRange();
    range.selectNodeContents(bibtex);
    selection.removeAllRanges();
    selection.addRange(range);
    if (label) label.textContent = 'Selected';
  }

  window.setTimeout(() => {
    if (label) label.textContent = 'Copy';
  }, 1800);
});
