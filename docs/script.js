const revealObserver = new IntersectionObserver((entries) => {
  entries.forEach((entry) => {
    if (entry.isIntersecting) {
      entry.target.classList.add('visible');
      revealObserver.unobserve(entry.target);
    }
  });
}, { threshold: 0.12 });

document.querySelectorAll('.reveal').forEach((element) => revealObserver.observe(element));

const copyButton = document.querySelector('[data-copy]');
copyButton?.addEventListener('click', async () => {
  const citation = document.querySelector('#bibtex').textContent;
  try {
    await navigator.clipboard.writeText(citation);
    copyButton.textContent = 'Copied';
    window.setTimeout(() => { copyButton.textContent = 'Copy'; }, 1800);
  } catch {
    copyButton.textContent = 'Select text';
  }
});
