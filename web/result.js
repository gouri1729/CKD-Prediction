const result = JSON.parse(localStorage.getItem('ckdResult') || 'null');
const data   = JSON.parse(localStorage.getItem('ckdData')   || '{}');

if (!result) {
  document.getElementById('resultBanner').innerHTML =
    '<p style="color:#f87171">No prediction data found. <a href="home.html">Go back</a></p>';
} else {
  const pred     = result.prediction;
  const prob     = result.probability;
  const risk     = result.risk_level;
  const allProbs = result.all_probabilities || {};

  // Color coding based on risk level (3-class system)
  const isHealthy = pred === 'No Disease/Healthy' || risk === 'Healthy';
  const isSevere  = pred === 'Has Disease' || risk === 'Has Disease';
  const isRisky   = pred === 'Risky' || risk === 'At Risk';

  const banner = document.getElementById('resultBanner');
  if (isHealthy) {
    banner.classList.add('result-healthy');
  } else if (isSevere) {
    banner.classList.add('result-ckd');
  } else {
    banner.classList.add('result-moderate');
  }

  // Icon selection
  const icon = isHealthy ? '✅' : (isSevere ? '⚠️' : '⚡');
  
  document.getElementById('resultIcon').textContent  = icon;
  document.getElementById('resultLabel').textContent = pred;
  document.getElementById('resultProb').textContent  = `Confidence: ${prob}%`;
  document.getElementById('resultRisk').textContent  = risk;

  // Gauge (doughnut) chart - showing predicted class probability
  const gaugeColor = isHealthy ? '#4ade80' : (isSevere ? '#f87171' : '#fbbf24');
  
  const ctx = document.getElementById('gaugeChart').getContext('2d');
  new Chart(ctx, {
    type: 'doughnut',
    data: {
      datasets: [{
        data: [prob, 100 - prob],
        backgroundColor: [gaugeColor, '#1e293b'],
        borderWidth: 0,
        circumference: 180,
        rotation: 270,
      }]
    },
    options: {
      cutout: '75%',
      plugins: {
        legend: { display: false },
        tooltip: { enabled: false }
      }
    }
  });

  // Center label
  Chart.register({
    id: 'centerText',
    afterDraw(chart) {
      const { ctx: c, width, height } = chart;
      c.save();
      c.font = 'bold 22px Segoe UI';
      c.fillStyle = '#e2e8f0';
      c.textAlign = 'center';
      c.fillText(prob + '%', width / 2, height - 10);
      c.restore();
    }
  });

  // Display all class probabilities (3-class system)
  const probEl = document.getElementById('allProbabilities');
  const classOrder = ['No Disease/Healthy', 'Risky', 'Has Disease'];
  
  classOrder.forEach(className => {
    if (allProbs[className] !== undefined) {
      const p = allProbs[className];
      const isSelected = className === pred;
      probEl.innerHTML += `
        <div class="prob-item ${isSelected ? 'prob-selected' : ''}">
          <span class="prob-name">${className}</span>
          <div class="prob-bar-container">
            <div class="prob-bar" style="width: ${p}%"></div>
          </div>
          <span class="prob-value">${p}%</span>
        </div>`;
    }
  });

  // Value summary table (10 features)
  const labels = {
    age:              'Age',
    blood_pressure:   'Blood Pressure',
    glucose:          'Blood Glucose',
    blood_urea:       'Blood Urea',
    serum_creatinine: 'Serum Creatinine',
    sodium:           'Sodium',
    potassium:        'Potassium',
    hemoglobin:       'Hemoglobin',
    egfr:             'eGFR',
    upcr:             'UPCR',
  };

  const summaryEl = document.getElementById('valueSummary');
  for (const [key, label] of Object.entries(labels)) {
    if (data[key] !== undefined) {
      summaryEl.innerHTML += `
        <div class="summary-item">
          <span class="summary-label">${label}</span>
          <span class="summary-value">${data[key]}</span>
        </div>`;
    }
  }
  
  // Fetch LIME explanations
  fetchLimeExplanations(data);
}

// Function to fetch and display LIME explanations
async function fetchLimeExplanations(data) {
  try {
    const response = await fetch('/explain', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data)
    });
    
    if (!response.ok) throw new Error('Failed to generate explanations');
    
    const limeData = await response.json();
    
    // Hide loading, show content
    document.getElementById('limeLoading').style.display = 'none';
    document.getElementById('limeContent').style.display = 'block';
    
    // Display plots
    if (limeData.plots.probability_chart) {
      document.getElementById('limeProbabilityChart').src = limeData.plots.probability_chart;
    }
    if (limeData.plots.feature_importance) {
      document.getElementById('limeFeaturePlot').src = limeData.plots.feature_importance;
    }
    
    // Display feature importance table
    const featureImportanceEl = document.getElementById('featureImportance');
    limeData.feature_importance.slice(0, 5).forEach((item, index) => {
      const impactColor = item.weight > 0 ? '#4ade80' : '#f87171';
      const impactIcon = item.weight > 0 ? '↑' : '↓';
      const absValue = Math.abs(item.weight).toFixed(4);
      
      featureImportanceEl.innerHTML += `
        <div class="feature-item">
          <div class="feature-rank">${index + 1}</div>
          <div class="feature-details">
            <div class="feature-name">${item.feature}</div>
            <div class="feature-value">Condition: ${item.condition}</div>
          </div>
          <div class="feature-impact" style="color: ${impactColor}">
            ${impactIcon} ${absValue}
          </div>
        </div>`;
    });
    
  } catch (error) {
    console.error('LIME error:', error);
    document.getElementById('limeLoading').style.display = 'none';
    document.getElementById('limeError').style.display = 'block';
  }
}

// ------------------------------------------------------------------
// AI Report Generation
// ------------------------------------------------------------------

async function generateFullReport() {
  const generateBtn = document.getElementById('generateReportBtn');
  const reportLoading = document.getElementById('reportLoading');
  const reportSection = document.getElementById('fullReportSection');
  const reportError = document.getElementById('reportError');
  
  // Hide previous results
  reportSection.style.display = 'none';
  reportError.style.display = 'none';
  
  // Show loading state
  generateBtn.disabled = true;
  generateBtn.style.opacity = '0.5';
  reportLoading.style.display = 'flex';
  
  try {
    const response = await fetch('/generate_report', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data)
    });
    
    const result = await response.json();
    
    if (result.success && result.report) {
      // Convert markdown-style formatting to HTML
      const htmlReport = formatReportToHTML(result.report);
      reportSection.innerHTML = htmlReport;
      reportSection.style.display = 'block';
      
      // Smooth scroll to report
      setTimeout(() => {
        reportSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
      }, 100);
      
      // Hide button after successful generation
      generateBtn.textContent = '✅ Report Generated';
      setTimeout(() => {
        generateBtn.style.display = 'none';
      }, 2000);
      
    } else {
      reportError.textContent = '⚠️ ' + (result.error || 'Unable to generate report. Please try again.');
      reportError.style.display = 'block';
      generateBtn.disabled = false;
      generateBtn.style.opacity = '1';
    }
  } catch (error) {
    console.error('Report generation failed:', error);
    reportError.textContent = '⚠️ Failed to generate report. Please check your internet connection and try again.';
    reportError.style.display = 'block';
    generateBtn.disabled = false;
    generateBtn.style.opacity = '1';
  } finally {
    reportLoading.style.display = 'none';
  }
}

function formatReportToHTML(text) {
  // Convert markdown-like formatting to HTML
  let html = text
    // Headers
    .replace(/^### (.*$)/gim, '<h3>$1</h3>')
    .replace(/^## (.*$)/gim, '<h2>$1</h2>')
    .replace(/^# (.*$)/gim, '<h1>$1</h1>')
    // Bold and italic
    .replace(/\*\*\*([^*]+)\*\*\*/g, '<strong><em>$1</em></strong>')
    .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
    .replace(/\*([^*]+)\*/g, '<em>$1</em>')
    // Bullet lists
    .replace(/^[-•]\s+(.*)$/gim, '<li>$1</li>')
    // Numbered lists
    .replace(/^\d+\.\s+(.*)$/gim, '<li>$1</li>')
    // Line breaks
    .replace(/\n\n/g, '<br><br>')
    .replace(/\n/g, '<br>');
  
  // Wrap consecutive <li> in <ul>
  html = html.replace(/(<li>.*?<\/li>(?:<br>)?)+/gs, function(match) {
    const items = match.replace(/<br>/g, '');
    return '<ul>' + items + '</ul>';
  });
  
  // Highlight disclaimer
  html = html.replace(/(⚕️.*?DISCLAIMER:.*?)(<br>|<h|$)/gi, 
    '<div class="disclaimer">$1</div>$2');
  
  return html;
}