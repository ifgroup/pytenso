/* ============================================================
   TENSO Search — search.js
   Client-side full-text search using pre-built JSON index
   ============================================================ */

'use strict';

const Search = (() => {
  let index = [];
  let loaded = false;

  async function load() {
    if (loaded) return;
    // Prefer inline index (works with file:// and avoids fetch CORS issues)
    if (window.__SEARCH_INDEX__) {
      index = window.__SEARCH_INDEX__;
      loaded = true;
      return;
    }
    try {
      const base = document.querySelector('base')?.href || window.location.href;
      const url = new URL('search-index.json', base).href;
      const res = await fetch(url);
      if (!res.ok) throw new Error('fetch failed');
      index = await res.json();
      loaded = true;
    } catch {
      console.warn('Search index not available');
    }
  }

  function query(q) {
    if (!q || q.length < 2) return [];
    const terms = q.toLowerCase().trim().split(/\s+/);

    return index
      .map(entry => {
        const text = (entry.title + ' ' + entry.content + ' ' + (entry.section || '')).toLowerCase();
        const score = terms.reduce((acc, term) => {
          if (entry.title.toLowerCase().includes(term)) return acc + 3;
          if ((entry.section || '').toLowerCase().includes(term)) return acc + 2;
          if (text.includes(term)) return acc + 1;
          return acc;
        }, 0);
        return { ...entry, score };
      })
      .filter(e => e.score > 0)
      .sort((a, b) => b.score - a.score)
      .slice(0, 8);
  }

  function renderResults(results, container) {
    container.innerHTML = '';
    if (!results.length) {
      container.innerHTML = '<div class="search-result-item" style="cursor:default"><div class="search-result-title">No results found</div></div>';
      return;
    }

    results.forEach(r => {
      const a = document.createElement('a');
      a.className = 'search-result-item';
      a.href = r.url;
      a.innerHTML = `
        <div class="search-result-title">${r.title}</div>
        <div class="search-result-section">${r.section || r.page}</div>
      `;
      container.appendChild(a);
    });
  }

  function init() {
    const input = document.querySelector('.search-input');
    const resultsEl = document.querySelector('.search-results');
    if (!input || !resultsEl) return;

    // Load index when input is focused
    input.addEventListener('focus', () => {
      load();
      if (input.value.length >= 2) {
        resultsEl.classList.add('search-results--open');
      }
    });

    let debounce;
    input.addEventListener('input', () => {
      clearTimeout(debounce);
      debounce = setTimeout(async () => {
        const q = input.value.trim();
        if (q.length < 2) {
          resultsEl.classList.remove('search-results--open');
          return;
        }
        await load();
        const results = query(q);
        renderResults(results, resultsEl);
        resultsEl.classList.add('search-results--open');
      }, 200);
    });

    // Keyboard navigation
    input.addEventListener('keydown', e => {
      if (e.key === 'Escape') {
        resultsEl.classList.remove('search-results--open');
        input.blur();
      }
      if (e.key === 'ArrowDown') {
        const first = resultsEl.querySelector('.search-result-item');
        if (first) first.focus();
        e.preventDefault();
      }
    });

    resultsEl.addEventListener('keydown', e => {
      const items = [...resultsEl.querySelectorAll('.search-result-item')];
      const idx = items.indexOf(document.activeElement);
      if (e.key === 'ArrowDown' && idx < items.length - 1) {
        items[idx + 1].focus(); e.preventDefault();
      }
      if (e.key === 'ArrowUp') {
        if (idx > 0) items[idx - 1].focus();
        else input.focus();
        e.preventDefault();
      }
      if (e.key === 'Escape') {
        resultsEl.classList.remove('search-results--open');
        input.focus();
      }
    });

    // Close on outside click
    document.addEventListener('click', e => {
      if (!input.contains(e.target) && !resultsEl.contains(e.target)) {
        resultsEl.classList.remove('search-results--open');
      }
    });
  }

  return { init, query, load };
})();

document.addEventListener('DOMContentLoaded', () => Search.init());
