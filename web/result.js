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
}
