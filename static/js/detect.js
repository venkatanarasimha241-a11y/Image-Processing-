document.addEventListener('DOMContentLoaded', () => {
  const dropzone = document.getElementById('dropzone');
  const imageInput = document.getElementById('imageInput');
  const fileName = document.getElementById('fileName');
  const preview = document.getElementById('preview');
  const form = document.querySelector('.upload-panel');

  if (!dropzone || !imageInput) return;

  function updatePreview(file) {
    if (!file) return;
    if (fileName) {
      fileName.textContent = `${file.name} (${(file.size / 1024 / 1024).toFixed(2)} MB)`;
    }

    if (file.type.startsWith('image/')) {
      const reader = new FileReader();
      reader.onload = (e) => {
        if (preview) {
          preview.innerHTML = `<img src="${e.target.result}" alt="Preview of ${file.name}">`;
        }
      };
      reader.readAsDataURL(file);
    }
  }

  imageInput.addEventListener('change', () => {
    if (imageInput.files && imageInput.files[0]) {
      updatePreview(imageInput.files[0]);
    }
  });

  ['dragenter', 'dragover'].forEach(eventName => {
    dropzone.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropzone.classList.add('dragover');
    });
  });

  ['dragleave', 'drop'].forEach(eventName => {
    dropzone.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropzone.classList.remove('dragover');
    });
  });

  dropzone.addEventListener('drop', (e) => {
    const dt = e.dataTransfer;
    const files = dt.files;
    if (files && files.length > 0) {
      imageInput.files = files;
      updatePreview(files[0]);
    }
  });

  if (form) {
    form.addEventListener('submit', () => {
      const btn = form.querySelector('button[type="submit"]');
      if (btn) {
        btn.textContent = 'Processing AI Inference...';
        btn.disabled = true;
        btn.style.opacity = '0.75';
      }
    });
  }
});
