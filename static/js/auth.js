document.addEventListener('DOMContentLoaded', () => {
  const tabs = document.querySelectorAll('.tabs button');
  const loginForm = document.getElementById('loginForm');
  const signupForm = document.getElementById('signupForm');
  const authTitle = document.getElementById('authTitle');
  const authSub = document.getElementById('authSub');

  tabs.forEach(tab => {
    tab.addEventListener('click', () => {
      const mode = tab.dataset.tab;
      tabs.forEach(t => t.classList.remove('active'));
      tab.classList.add('active');

      if (mode === 'signup') {
        loginForm.classList.add('hidden');
        signupForm.classList.remove('hidden');
        if (authTitle) authTitle.textContent = 'Create Account';
        if (authSub) authSub.textContent = 'Register for your secure AI Vision workspace';
      } else {
        signupForm.classList.add('hidden');
        loginForm.classList.remove('hidden');
        if (authTitle) authTitle.textContent = 'Welcome Back';
        if (authSub) authSub.textContent = 'Sign in to your account to continue';
      }
    });
  });

  // Password visibility toggle
  const passButtons = document.querySelectorAll('button[data-password]');
  passButtons.forEach(btn => {
    btn.addEventListener('click', () => {
      const input = btn.parentElement.querySelector('input');
      if (!input) return;
      if (input.type === 'password') {
        input.type = 'text';
        btn.textContent = '◎';
      } else {
        input.type = 'password';
        btn.textContent = '◉';
      }
    });
  });
});
