/* Hub tag filter — multi-select, OR logic. No dependencies. */
(function () {
  var bar = document.querySelector('[data-filters]');
  if (!bar) return;

  var cards = [].slice.call(document.querySelectorAll('.entry[data-tags]'));
  var count = document.querySelector('[data-count]');
  var empty = document.querySelector('[data-empty]');
  var buttons = [].slice.call(bar.querySelectorAll('.filter'));
  var active = [];

  function render() {
    var shown = 0;
    cards.forEach(function (card) {
      var tags = (card.getAttribute('data-tags') || '').split(',');
      var match = active.length === 0 || active.some(function (t) {
        return tags.indexOf(t) !== -1;
      });
      card.hidden = !match;
      if (match) shown++;
    });

    buttons.forEach(function (b) {
      var t = b.getAttribute('data-tag');
      b.setAttribute('aria-pressed',
        t === '*' ? String(active.length === 0) : String(active.indexOf(t) !== -1));
    });

    if (empty) empty.hidden = shown !== 0;
    if (count) {
      count.textContent = active.length === 0
        ? 'Showing all ' + cards.length + ' sessions'
        : 'Showing ' + shown + ' of ' + cards.length + ' sessions · ' + active.join(', ');
    }
  }

  bar.addEventListener('click', function (e) {
    var btn = e.target.closest ? e.target.closest('.filter') : null;
    if (!btn) return;
    var tag = btn.getAttribute('data-tag');
    if (tag === '*') {
      active = [];
    } else {
      var i = active.indexOf(tag);
      if (i === -1) active.push(tag); else active.splice(i, 1);
    }
    render();
  });

  render();
})();
