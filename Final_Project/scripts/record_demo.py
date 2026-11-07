"""Development-only browser capture. Start Streamlit on port 8501 first.
Requires playwright and its ffmpeg encoder; uses system Google Chrome.
"""
from pathlib import Path
import json
from playwright.sync_api import sync_playwright
ROOT=Path(__file__).resolve().parents[1]
out=ROOT/'demo/presentation';out.mkdir(exist_ok=True)
scenes=[]
with sync_playwright() as p:
    browser=p.chromium.launch(executable_path='/usr/bin/google-chrome',headless=True,args=['--no-sandbox','--disable-dev-shm-usage'])
    context=browser.new_context(viewport={'width':1440,'height':1100},record_video_dir='/tmp/bea-demo-video',record_video_size={'width':1440,'height':1100})
    page=context.new_page();page.goto('http://127.0.0.1:8501')
    page.locator('.js-plotly-plot').first.wait_for(timeout=60000)
    page.wait_for_function("document.querySelectorAll('[data-testid=stImage] img').length === 4 && [...document.querySelectorAll('[data-testid=stImage] img')].every(i=>i.complete && i.naturalWidth > 0)",timeout=60000)
    def scene(name,seconds):
        page.wait_for_timeout(1000)
        assert not page.locator('[data-testid=stException]').count()
        scenes.append(dict(scene=name,cards=page.locator('[data-testid=stMetric]').all_text_contents()))
        page.wait_for_timeout(seconds*1000)
    scene('Clean / clean-trained / ensemble 01 / 90%',4)
    page.get_by_role('combobox').first.click();page.get_by_role('option',name='Gaussian noise',exact=True).click()
    scene('Gaussian noise 3 / clean-trained / ensemble 01 / 90%',4)
    page.get_by_text('Noise-adapted',exact=True).click()
    scene('Gaussian noise 3 / noise-adapted / ensemble 01 / 90%',4)
    page.get_by_role('combobox').first.click();page.get_by_role('option',name='Motion blur',exact=True).click()
    scene('Motion blur 3 / noise-adapted / ensemble 01 / 90%',4)
    page.get_by_role('tab',name='Replication evidence').click();page.wait_for_timeout(4000)
    video=page.video;context.close();video.save_as(str(out/'backup.webm'));browser.close()
(out/'recording_checks.json').write_text(json.dumps(scenes,indent=2))
print(json.dumps(scenes,indent=2))
