# CKD Prediction System with LLM Integration 🤖

## Overview
This enhanced CKD (Chronic Kidney Disease) prediction system now includes AI-powered report generation using OpenAI's GPT models. The system provides:

1. **Machine Learning Predictions** - Uses Random Forest with SMOTE and feature selection
2. **LIME Explanations** - Visual explanations of model decisions
3. **AI-Generated Reports** - Comprehensive, patient-friendly medical reports

## Features ✨

### 1. Prediction System
- 3-class classification (No Disease / At Risk / Has Disease)
- Top 10 RFE-selected clinical features
- Confidence scores and probability breakdown

### 2. LIME Explanations
- Feature importance visualization
- Interactive charts showing which factors influenced the prediction
- Color-coded risk/protective factors

### 3. AI-Powered Reports (NEW! 🆕)
- **Patient-Friendly Language** - Explains medical terms in simple English
- **Personalized Analysis** - Tailored to individual test results
- **Risk Factor Breakdown** - Clear explanation of what's causing the diagnosis
- **Actionable Recommendations** - Specific lifestyle and dietary advice
- **Medical Guidance** - When to see a doctor, what questions to ask
- **Timeline** - Step-by-step action plan (24 hours, 1 week, 1 month)

## Setup Instructions

### 1. Install Dependencies
```bash
pip install openai python-dotenv
```

Or install all requirements:
```bash
pip install -r requirements_llm.txt
```

### 2. Get OpenAI API Key
1. Visit https://platform.openai.com/api-keys
2. Create an account or log in
3. Create a new API key
4. Copy the key (starts with `sk-...`)

### 3. Configure Environment Variables
Edit the `.env` file in the project root:
```bash
OPENAI_API_KEY=sk-your-actual-api-key-here
OPENAI_MODEL=gpt-3.5-turbo
```

**Models Available:**
- `gpt-3.5-turbo` - Fast and cost-effective ($0.50 per 1M tokens)
- `gpt-4-turbo-preview` - More accurate, higher cost
- `gpt-4` - Best quality, highest cost

### 4. Run the Application
```bash
python llm.py
```

Then open: **http://127.0.0.1:5000**

## How It Works

### User Flow:
1. **Input Clinical Data** - Enter 10 key health indicators
2. **Get Prediction** - See AI classification and confidence
3. **View LIME Explanation** - Understand which factors matter most
4. **Generate AI Report** - Click button to get comprehensive analysis
5. **Read Personalized Report** - Get clear, actionable health advice

### LLM Integration:
The system sends structured data to OpenAI's GPT:
- Patient test results with normal ranges
- Abnormal values flagged by severity
- LIME-identified risk factors
- AI model's prediction and confidence

GPT then generates:
- **Summary** - Quick health status overview
- **Results Explained** - Each abnormal test in plain language
- **Risk Analysis** - Key factors affecting kidney health
- **Action Plan** - Immediate actions, appointments, lifestyle changes
- **Questions for Doctor** - Specific questions to ask based on results

## Report Sections

### 1. Summary (2-3 sentences)
Clear verdict on kidney health status and urgency level.

### 2. Understanding Your Results
Patient-friendly explanation of the diagnosis without medical jargon.

### 3. Your Test Results Explained
For each abnormal value:
- What it measures
- Your value vs normal range
- Why it matters
- Potential causes

### 4. Key Risk Factors
AI-identified factors from LIME analysis explained in context.

### 5. Action Plan
- **Immediate** - Critical actions within 24 hours
- **Short-term** - Appointments and tests within 1 week
- **Lifestyle** - Specific diet, exercise, and habit changes
- **Monitoring** - Follow-up schedule

### 6. Positive Aspects
Acknowledges normal test results and encouraging factors.

### 7. Next Steps Timeline
Structured action items with timeframes.

### 8. Questions for Your Doctor
5-7 specific questions based on individual results.

## Cost Estimation

**OpenAI API Costs** (using gpt-3.5-turbo):
- Average report: 1,500-2,500 tokens
- Cost: ~$0.001 - $0.002 per report
- 1,000 reports: ~$1-2

**Tips to Reduce Costs:**
1. Use `gpt-3.5-turbo` instead of `gpt-4`
2. Implement caching for similar cases
3. Set usage limits in OpenAI dashboard

## Error Handling

The system includes fallback mechanisms:
- If OpenAI API fails, generates structured text report
- Network errors show user-friendly messages
- Retry logic for temporary failures

## Privacy & Security

- No patient data is stored by OpenAI (API calls only)
- All data transmitted over HTTPS
- API keys stored in `.env` (not committed to git)
- Include `.env` in `.gitignore`

## Medical Disclaimer ⚕️

**IMPORTANT:** This is an AI-powered screening tool for educational purposes only. It is NOT a medical diagnosis and should NOT replace professional medical advice, diagnosis, or treatment.

Always consult with qualified healthcare providers for:
- Proper medical evaluation
- Diagnosis confirmation
- Treatment decisions
- Medical emergencies

If experiencing severe symptoms, seek immediate medical attention.

## Troubleshooting

### "OpenAI API Error"
- Check your API key is correct in `.env`
- Verify you have credits in your OpenAI account
- Check internet connection

### "Report generation failed"
- Ensure `python-dotenv` is installed
- Verify `.env` file is in the project root
- Check OpenAI API status: https://status.openai.com

### Model loads but no report generated
- Check browser console for JavaScript errors
- Verify `/generate_report` endpoint is responding
- Try clearing browser cache

## File Structure
```
CKD-Prediction/
├── .env                      # API keys (DO NOT COMMIT)
├── llm.py                    # Main Flask app with LLM integration
├── end.py                    # Model training script
├── ckd_model.pkl            # Trained model
├── model_metadata.json       # Feature information
├── requirements_llm.txt      # Dependencies
├── README_LLM.md            # This file
└── web/
    ├── home.html            # Input form
    ├── result.html          # Results page with report
    ├── result.js            # Frontend logic + LLM integration
    ├── script.js            # Form handling
    └── style.css            # Styling including report styles
```

## Future Enhancements

Potential additions:
- [ ] Multi-language report generation
- [ ] PDF export of reports
- [ ] Email report delivery
- [ ] Historical tracking and trends
- [ ] Integration with electronic health records
- [ ] Voice synthesis to read reports
- [ ] SMS alerts for critical findings

## Support

For issues or questions:
1. Check this README
2. Review OpenAI documentation
3. Check Flask documentation
4. Verify all dependencies are installed

## License

This project is for educational and research purposes.

---

**Built with ❤️ using Flask, Scikit-learn, LIME, and OpenAI GPT**
