const seqForm = document.getElementById("seq-form");
const smilesForm = document.getElementById("smiles-form");
const templateForm = document.getElementById("template-form");

const seqResult = document.getElementById("seq-result");
const smilesResult = document.getElementById("smiles-result");
const templateResult = document.getElementById("template-result");

function showResult(target, payload) {
  target.classList.remove("error");
  target.textContent = typeof payload === "string" ? payload : JSON.stringify(payload, null, 2);
}

function showError(target, message) {
  target.classList.add("error");
  target.textContent = message;
}

async function postJSON(url, data) {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });
  if (!response.ok) {
    const detail = await response.json().catch(() => ({}));
    const message = detail?.detail || `Request failed with status ${response.status}`;
    throw new Error(message);
  }
  return response.json();
}

seqForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const sequence = event.target.sequence.value.trim();
  if (!sequence) {
    showError(seqResult, "Please enter a sequence.");
    return;
  }
  showResult(seqResult, "Converting…");
  try {
    const data = await postJSON("/api/seq2smi", { sequence });
    showResult(seqResult, `SMILES: ${data.smiles}`);
  } catch (error) {
    showError(seqResult, error.message);
  }
});

smilesForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const smiles = event.target.smiles.value.trim();
  if (!smiles) {
    showError(smilesResult, "Please enter a SMILES string.");
    return;
  }
  showResult(smilesResult, "Converting…");
  try {
    const data = await postJSON("/api/smi2seq", { smiles });
    showResult(
      smilesResult,
      `Sequence: ${data.sequence}\n\nDetails:\n${JSON.stringify(data.details, null, 2)}`
    );
  } catch (error) {
    showError(smilesResult, error.message);
  }
});

templateForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const code = event.target.code.value.trim();
  const smiles = event.target.smiles.value.trim();
  const aliases = event.target.aliases.value
    .split(",")
    .map((item) => item.trim())
    .filter(Boolean);
  const components = event.target.components.value
    .split(",")
    .map((item) => item.trim())
    .filter(Boolean);

  if (!code || !smiles) {
    showError(templateResult, "Code and SMILES are required.");
    return;
  }
  showResult(templateResult, "Registering template…");
  try {
    const payload = { code, smiles };
    if (aliases.length) payload.aliases = aliases;
    if (components.length) payload.components = components;
    const data = await postJSON("/api/templates", payload);
    showResult(templateResult, `Success: ${JSON.stringify(data.entry, null, 2)}`);
    event.target.reset();
  } catch (error) {
    showError(templateResult, error.message);
  }
});
