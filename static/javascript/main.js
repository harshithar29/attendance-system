// Global front-end interactions for the AI Attendance System.
// The login experience uses a small amount of UI polish to keep the flow smooth.

document.addEventListener("DOMContentLoaded", () => {
    const passwordInput = document.getElementById("password");
    const passwordToggle = document.querySelector(".password-toggle");

    if (passwordInput && passwordToggle) {
        passwordToggle.addEventListener("click", () => {
            const isHidden = passwordInput.type === "password";
            passwordInput.type = isHidden ? "text" : "password";

            const icon = passwordToggle.querySelector("i");
            icon.className = isHidden ? "bi bi-eye-slash" : "bi bi-eye";
            passwordToggle.setAttribute("aria-label", isHidden ? "Hide password" : "Show password");
        });
    }

    const loginForm = document.getElementById("loginForm");
    const loginButton = document.getElementById("loginButton");

    if (loginForm && loginButton) {
        loginForm.addEventListener("submit", () => {
            const label = loginButton.querySelector(".btn-label");
            const spinner = loginButton.querySelector(".btn-spinner");

            label.classList.add("d-none");
            spinner.classList.remove("d-none");
            loginButton.disabled = true;
        });
    }
});
