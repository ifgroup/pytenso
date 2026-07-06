/* ============================================================
   TENSO Website — main.js
   All JavaScript modules
   ============================================================ */

'use strict';

/* ============================================================
   ThemeManager
   ============================================================ */
const ThemeManager = (() => {
  const STORAGE_KEY = 'tenso-theme';
  const root = document.documentElement;

  function getTheme() {
    return localStorage.getItem(STORAGE_KEY) ||
      (window.matchMedia('(prefers-color-scheme: light)').matches ? 'light' : 'dark');
  }

  function apply(theme) {
    root.setAttribute('data-theme', theme);
    document.querySelectorAll('.theme-toggle-icon').forEach(icon => {
      icon.textContent = theme === 'dark' ? '☀️' : '🌙';
    });
    localStorage.setItem(STORAGE_KEY, theme);
  }

  function toggle() {
    apply(root.getAttribute('data-theme') === 'dark' ? 'light' : 'dark');
  }

  function init() {
    apply(getTheme());
    document.querySelectorAll('[data-theme-toggle]').forEach(btn => {
      btn.addEventListener('click', toggle);
    });
  }

  return { init, toggle };
})();

/* ============================================================
   NavManager
   ============================================================ */
const NavManager = (() => {
  function init() {
    const nav = document.querySelector('.nav');
    const hamburger = document.querySelector('.hamburger');
    const links = document.querySelector('.nav__links');

    if (!nav) return;

    // Scroll state
    window.addEventListener('scroll', () => {
      nav.classList.toggle('nav--scrolled', window.scrollY > 20);
    }, { passive: true });

    // Hamburger toggle
    if (hamburger && links) {
      hamburger.addEventListener('click', () => {
        const open = links.classList.toggle('nav__links--open');
        hamburger.setAttribute('aria-expanded', open);
      });
    }

    // Active link detection
    const current = window.location.pathname.split('/').pop() || 'index.html';
    document.querySelectorAll('.nav__link').forEach(link => {
      const href = link.getAttribute('href');
      if (href === current || (current === '' && href === 'index.html')) {
        link.classList.add('nav__link--active');
      }
    });
  }

  return { init };
})();

/* ============================================================
   ProgressBar
   ============================================================ */
const ProgressBar = (() => {
  function init() {
    const bar = document.querySelector('.progress-bar');
    if (!bar) return;

    function update() {
      const scrolled = window.scrollY;
      const height = document.documentElement.scrollHeight - window.innerHeight;
      const progress = height > 0 ? scrolled / height : 0;
      document.documentElement.style.setProperty('--scroll-progress', progress);
      requestAnimationFrame(update);
    }

    requestAnimationFrame(update);
  }

  return { init };
})();

/* ============================================================
   RevealManager
   ============================================================ */
const RevealManager = (() => {
  function init() {
    const els = document.querySelectorAll('.reveal');
    if (!els.length) return;

    const observer = new IntersectionObserver((entries) => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          entry.target.classList.add('reveal--visible');
          observer.unobserve(entry.target);
        }
      });
    }, { threshold: 0.1, rootMargin: '0px 0px -50px 0px' });

    els.forEach(el => observer.observe(el));
  }

  return { init };
})();

/* ============================================================
   CopyManager
   ============================================================ */
const CopyManager = (() => {
  function attachTo(el) {
    const btn = document.createElement('button');
    btn.className = 'copy-btn';
    btn.textContent = 'Copy';
    btn.setAttribute('aria-label', 'Copy code');

    btn.addEventListener('click', async () => {
      const code = el.querySelector('code') || el;
      try {
        await navigator.clipboard.writeText(code.textContent);
        btn.textContent = 'Copied!';
        btn.style.color = 'var(--accent-tertiary)';
        setTimeout(() => {
          btn.textContent = 'Copy';
          btn.style.color = '';
        }, 2000);
      } catch {
        btn.textContent = 'Failed';
        setTimeout(() => { btn.textContent = 'Copy'; }, 2000);
      }
    });

    // Wrap in code-block if needed
    if (!el.closest('.code-block')) {
      el.style.position = 'relative';
    }
    el.appendChild(btn);
  }

  function init() {
    document.querySelectorAll('pre:not(.no-copy)').forEach(pre => {
      const wrapper = pre.closest('.code-block') || pre;
      if (!wrapper.querySelector('.copy-btn')) {
        attachTo(wrapper === pre ? pre : wrapper);
      }
    });
  }

  return { init };
})();

/* ============================================================
   TabManager — generic ARIA tablist
   ============================================================ */
const TabManager = (() => {
  function initTablist(tablist) {
    const tabs = [...tablist.querySelectorAll('[role="tab"]')];
    const panels = tabs.map(t => document.getElementById(t.getAttribute('aria-controls')));

    function activate(tab) {
      tabs.forEach((t, i) => {
        const selected = t === tab;
        t.setAttribute('aria-selected', selected);
        t.tabIndex = selected ? 0 : -1;
        if (panels[i]) panels[i].setAttribute('aria-hidden', !selected);
      });
    }

    tabs.forEach(tab => {
      tab.addEventListener('click', () => activate(tab));
      tab.addEventListener('keydown', e => {
        const idx = tabs.indexOf(document.activeElement);
        if (e.key === 'ArrowRight') tabs[(idx + 1) % tabs.length].focus();
        if (e.key === 'ArrowLeft') tabs[(idx - 1 + tabs.length) % tabs.length].focus();
        if (e.key === 'Home') tabs[0].focus();
        if (e.key === 'End') tabs[tabs.length - 1].focus();
      });
    });

    // Activate first
    const active = tabs.find(t => t.getAttribute('aria-selected') === 'true') || tabs[0];
    if (active) activate(active);
  }

  function init() {
    document.querySelectorAll('[role="tablist"]').forEach(initTablist);
  }

  return { init };
})();

/* ============================================================
   OSTabManager — OS tab switcher
   ============================================================ */
const OSTabManager = (() => {
  function init() {
    document.querySelectorAll('.os-tabs').forEach(container => {
      const btns = [...container.querySelectorAll('.os-tab-btn')];
      if (!btns.length) return;

      const panelId = container.dataset.panels;
      const panelContainer = panelId ? document.getElementById(panelId) :
        container.nextElementSibling;
      if (!panelContainer) return;

      const panels = [...panelContainer.querySelectorAll('.os-panel')];

      function activate(btn) {
        const target = btn.dataset.os;
        btns.forEach(b => {
          b.classList.toggle('active', b === btn);
          b.setAttribute('aria-selected', b === btn);
        });
        panels.forEach(p => {
          p.classList.toggle('active', p.dataset.os === target);
        });
      }

      btns.forEach(btn => {
        btn.addEventListener('click', () => activate(btn));
        btn.addEventListener('keydown', e => {
          const idx = btns.indexOf(document.activeElement);
          if (e.key === 'ArrowRight') { btns[(idx + 1) % btns.length].focus(); e.preventDefault(); }
          if (e.key === 'ArrowLeft') { btns[(idx - 1 + btns.length) % btns.length].focus(); e.preventDefault(); }
        });
      });

      activate(btns[0]);
    });
  }

  return { init };
})();

/* ============================================================
   TOCManager — highlight active TOC link
   ============================================================ */
const TOCManager = (() => {
  function init() {
    const toc = document.querySelector('.toc');
    if (!toc) return;

    const links = [...toc.querySelectorAll('.toc__link')];
    if (!links.length) return;

    const targets = links.map(link => {
      const id = link.getAttribute('href')?.replace('#', '');
      return { link, el: id ? document.getElementById(id) : null };
    }).filter(t => t.el);

    if (!targets.length) return;

    // For OS tab panels: intercept TOC clicks to activate the tab first
    targets.forEach(({ link, el }) => {
      if (el.classList.contains('os-panel')) {
        link.addEventListener('click', e => {
          const btn = document.querySelector(`.os-tab-btn[data-os="${el.dataset.os}"]`);
          if (btn && !el.classList.contains('active')) {
            e.preventDefault();
            btn.click();
            setTimeout(() => el.scrollIntoView({ behavior: 'smooth', block: 'start' }), 50);
          }
        });
      }
    });

    function update() {
      const threshold = window.scrollY + window.innerHeight * 0.3;
      let active = targets[0];
      for (const t of targets) {
        if (t.el.offsetParent === null) continue; // skip hidden elements
        if (t.el.offsetTop <= threshold) active = t;
      }
      links.forEach(l => l.classList.remove('toc__link--active'));
      if (active) active.link.classList.add('toc__link--active');
    }

    window.addEventListener('scroll', update, { passive: true });
    update();
  }

  return { init };
})();

/* ============================================================
   ParticleSystem — canvas animation
   ============================================================ */
const ParticleSystem = (() => {
  const PARTICLE_COUNT = 200;
  const CONNECT_DIST = 100;
  const MOUSE_REPEL = 80;

  class Particle {
    constructor(w, h) {
      this.reset(w, h);
    }
    reset(w, h) {
      this.x = Math.random() * w;
      this.y = Math.random() * h;
      this.vx = (Math.random() - 0.5) * 0.4;
      this.vy = (Math.random() - 0.5) * 0.4;
      this.phase = Math.random() * Math.PI * 2;
      this.size = Math.random() * 2 + 0.5;
      this.alpha = Math.random() * 0.5 + 0.2;
    }
    update(w, h, t, mouse) {
      this.x += this.vx + Math.sin(t * 0.001 + this.phase) * 0.3;
      this.y += this.vy + Math.cos(t * 0.0013 + this.phase) * 0.3;

      // Mouse repulsion
      if (mouse.x !== null) {
        const dx = this.x - mouse.x;
        const dy = this.y - mouse.y;
        const dist = Math.sqrt(dx * dx + dy * dy);
        if (dist < MOUSE_REPEL) {
          const force = (MOUSE_REPEL - dist) / MOUSE_REPEL;
          this.x += (dx / dist) * force * 2;
          this.y += (dy / dist) * force * 2;
        }
      }

      // Wrap
      if (this.x < -10) this.x = w + 10;
      if (this.x > w + 10) this.x = -10;
      if (this.y < -10) this.y = h + 10;
      if (this.y > h + 10) this.y = -10;
    }
  }

  function init() {
    const canvas = document.getElementById('hero-canvas');
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    let particles = [];
    let animId;
    let t = 0;
    const mouse = { x: null, y: null };

    function resize() {
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight;
    }

    function createParticles() {
      particles = Array.from({ length: PARTICLE_COUNT },
        () => new Particle(canvas.width, canvas.height));
    }

    function getColor() {
      return document.documentElement.getAttribute('data-theme') === 'light'
        ? '0,119,182' : '0,212,255';
    }

    function draw() {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      t++;

      const color = getColor();

      // Draw connections
      for (let i = 0; i < particles.length; i++) {
        for (let j = i + 1; j < particles.length; j++) {
          const dx = particles[i].x - particles[j].x;
          const dy = particles[i].y - particles[j].y;
          const dist = Math.sqrt(dx * dx + dy * dy);
          if (dist < CONNECT_DIST) {
            const alpha = (1 - dist / CONNECT_DIST) * 0.15;
            ctx.beginPath();
            ctx.strokeStyle = `rgba(${color},${alpha})`;
            ctx.lineWidth = 0.5;
            ctx.moveTo(particles[i].x, particles[i].y);
            ctx.lineTo(particles[j].x, particles[j].y);
            ctx.stroke();
          }
        }
      }

      // Draw particles
      particles.forEach(p => {
        p.update(canvas.width, canvas.height, t, mouse);
        ctx.beginPath();
        ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(${color},${p.alpha})`;
        ctx.fill();
      });

      animId = requestAnimationFrame(draw);
    }

    // Mouse tracking on window (canvas is full-page fixed)
    window.addEventListener('mousemove', e => {
      mouse.x = e.clientX;
      mouse.y = e.clientY;
    });
    window.addEventListener('mouseleave', () => {
      mouse.x = null; mouse.y = null;
    });

    // Touch support
    window.addEventListener('touchmove', e => {
      const touch = e.touches[0];
      mouse.x = touch.clientX;
      mouse.y = touch.clientY;
    }, { passive: true });

    window.addEventListener('resize', () => {
      resize();
      createParticles();
    });

    resize();
    createParticles();
    draw();
  }

  return { init };
})();

/* ============================================================
   TypewriterEffect
   ============================================================ */
const TypewriterEffect = (() => {
  const phrases = [
    'Exact open quantum dynamics',
    'Polynomial scaling via TTN-HEOM',
    'Structured thermal environments',
    'Tree tensor network propagation',
    'Ready for your quantum system'
  ];

  function init() {
    const el = document.getElementById('typewriter-text');
    if (!el) return;

    let pi = 0, ci = 0, deleting = false, paused = false;

    function tick() {
      const phrase = phrases[pi];

      if (paused) {
        paused = false;
        setTimeout(tick, 1500);
        return;
      }

      if (!deleting) {
        el.textContent = phrase.slice(0, ci + 1);
        ci++;
        if (ci === phrase.length) {
          deleting = true;
          paused = true;
          setTimeout(tick, 50);
          return;
        }
        setTimeout(tick, 60);
      } else {
        el.textContent = phrase.slice(0, ci - 1);
        ci--;
        if (ci === 0) {
          deleting = false;
          pi = (pi + 1) % phrases.length;
          setTimeout(tick, 300);
          return;
        }
        setTimeout(tick, 35);
      }
    }

    tick();
  }

  return { init };
})();

/* ============================================================
   CollapsibleManager
   ============================================================ */
const CollapsibleManager = (() => {
  function init() {
    document.querySelectorAll('.collapsible').forEach(col => {
      const header = col.querySelector('.collapsible__header');
      const body = col.querySelector('.collapsible__body');
      const id = col.dataset.id;

      if (!header || !body) return;

      // Restore from sessionStorage
      if (id && sessionStorage.getItem(`col-${id}`) === 'open') {
        col.classList.add('collapsible--open');
        body.style.maxHeight = body.scrollHeight + 'px';
      }

      header.addEventListener('click', () => {
        const open = col.classList.toggle('collapsible--open');
        body.style.maxHeight = open ? body.scrollHeight + 'px' : '0';
        if (id) sessionStorage.setItem(`col-${id}`, open ? 'open' : 'closed');
      });
    });
  }

  return { init };
})();

/* ============================================================
   CitationManager
   ============================================================ */
const CitationManager = (() => {
  function init() {
    // Global format tabs switch all citation blocks
    const globalTabs = document.querySelectorAll('.global-format-tab');
    if (globalTabs.length) {
      globalTabs.forEach(btn => {
        btn.addEventListener('click', () => {
          const fmt = btn.dataset.format;
          globalTabs.forEach(b => b.classList.toggle('active', b === btn));

          document.querySelectorAll('.citation-block').forEach(block => {
            block.classList.toggle('active', block.dataset.format === fmt);
          });
        });
      });
    }

    // Per-card copy buttons
    document.querySelectorAll('.citation-copy-btn').forEach(btn => {
      btn.addEventListener('click', async () => {
        const card = btn.closest('.citation-card');
        const active = card ? card.querySelector('.citation-block.active pre') : null;
        const text = (active || btn.previousElementSibling)?.textContent || '';
        try {
          await navigator.clipboard.writeText(text);
          btn.textContent = 'Copied!';
          setTimeout(() => { btn.textContent = 'Copy'; }, 2000);
        } catch {
          btn.textContent = 'Failed';
          setTimeout(() => { btn.textContent = 'Copy'; }, 2000);
        }
      });
    });
  }

  return { init };
})();

/* ============================================================
   Boot
   ============================================================ */
document.addEventListener('DOMContentLoaded', () => {
  ThemeManager.init();
  NavManager.init();
  ProgressBar.init();
  RevealManager.init();
  TabManager.init();
  OSTabManager.init();
  TOCManager.init();
  CollapsibleManager.init();
  CitationManager.init();
  CopyManager.init();
  ParticleSystem.init();
  TypewriterEffect.init();
});
