const { chromium } = require('playwright');
(async () => {
  const browser = await chromium.launch();
  const page = await browser.newPage();
  await page.goto('https://github.com/OpenBMB/VoxCPM');
  const text = await page.innerText('body');
  console.log(text.substring(0, 1000));
  
  // let's grab specific section
  const apiSection = text.split(/API/i).map((s, i) => i + ":" + s.substring(0, 300)).join('\n\n');
  const fs = require('fs');
  fs.writeFileSync('api_info.txt', apiSection);
  await browser.close();
})();
