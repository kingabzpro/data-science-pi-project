# Webwright test plan: Adult Income Streamlit app

## Critical points

- [x] CP1: Streamlit app starts successfully and the homepage loads. (run_2, step 2, screenshot `final_execution_2_homepage.png`)
- [x] CP2: The prediction form renders with all input controls visible. (run_2, step 3, screenshot `final_execution_3_form_visible.png`)
- [x] CP3: Submitting the form with valid inputs triggers a prediction. (run_2, step 4-5, screenshots `final_execution_4_form_filled.png`, `final_execution_5_prediction.png`)
- [x] CP4: The result section displays a predicted class and class probabilities. (run_2, step 6-7, log line "Predicted income: >50K", screenshot `final_execution_6_probabilities.png`)
- [x] CP5: No severe console errors are logged during the main user journey. (run_2, `console_errors.json` is `[]`)
- [x] CP6: Screenshots and an action log are saved for the run. (run_2, `final_script_log.txt` + 5 PNGs)

## Verified run

`final_runs/run_2/` is the clean, verified run. Final response: **Predicted income: >50K**, severe console errors: **0**.
