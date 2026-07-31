import { test, expect } from "@playwright/test";

test.describe("Mental Health Triage Platform E2E", () => {
  test("loads landing page and starts session", async ({ page }) => {
    await page.goto("http://localhost:3000/");
    await expect(page).toHaveTitle(/Mental Health|Google/i);

    const startButton = page.locator("button", { hasText: /start|begin/i });
    if (await startButton.isVisible()) {
      await startButton.click();
      await page.waitForURL(/\/session\//);
      expect(page.url()).toContain("/session/");
    }
  });

  test("triggers panic mode on ESC key press", async ({ page }) => {
    await page.goto("http://localhost:3000/");
    await page.keyboard.press("Escape");
    await expect(page).toHaveTitle("Google");
    await expect(page.locator("text=How to make sourdough bread from scratch")).toBeVisible();
  });
});
