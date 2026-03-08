document.getElementById('ckdForm').addEventListener('submit', async function (e) {
  e.preventDefault();

  const errorBox = document.getElementById('formError');
  const btn      = document.getElementById('submitBtn');

  // Validate all required fields
  const fields = this.querySelectorAll('[required]');
  for (const f of fields) {
    if (!f.value || f.value.trim() === '') {
      errorBox.style.display = 'block';
      f.focus();
      return;
    }
  }
  errorBox.style.display = 'none';

  // Collect the top 10 feature values
  const data = {
    age:              parseFloat(document.getElementById('age').value),
    serum_creatinine: parseFloat(document.getElementById('serum_creatinine').value),
    blood_urea:       parseFloat(document.getElementById('blood_urea').value),
    egfr:             parseFloat(document.getElementById('egfr').value),
    upcr:             parseFloat(document.getElementById('upcr').value),
    blood_pressure:   parseFloat(document.getElementById('blood_pressure').value),
    glucose:          parseFloat(document.getElementById('glucose').value),
    hemoglobin:       parseFloat(document.getElementById('hemoglobin').value),
    sodium:           parseFloat(document.getElementById('sodium').value),
    potassium:        parseFloat(document.getElementById('potassium').value),
  };

  // Show loading state
  btn.disabled   = true;
  btn.textContent = 'Predicting...';

  try {
    const res = await fetch('/predict', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify(data),
    });

    if (!res.ok) throw new Error('Server error: ' + res.status);

    const result = await res.json();
    if (result.error) throw new Error(result.error);

    // Save for result page
    localStorage.setItem('ckdResult', JSON.stringify(result));
    localStorage.setItem('ckdData',   JSON.stringify(data));

    window.location.href = 'result.html';

  } catch (err) {
    alert('Prediction failed: ' + err.message);
    btn.disabled    = false;
    btn.textContent = 'Predict CKD Risk';
  }
});
