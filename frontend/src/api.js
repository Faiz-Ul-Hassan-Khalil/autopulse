const API_BASE =
  import.meta.env.VITE_API_URL ||
  `http://${window.location.hostname || "localhost"}:8000`;

export async function fetchOptions() {
  const res = await fetch(`${API_BASE}/options`);
  if (!res.ok) throw new Error("Failed to load car catalog from backend");
  return res.json();
}

export async function fetchModelInfo() {
  const res = await fetch(`${API_BASE}/model-info`);
  if (!res.ok) return { available: false };
  return res.json();
}

export async function predictCarPrice(payload) {
  const res = await fetch(`${API_BASE}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const data = await res.json();
  if (!res.ok) {
    const detail = Array.isArray(data.detail)
      ? data.detail.map((e) => e.msg).join(", ")
      : data.detail || "Prediction failed";
    throw new Error(detail);
  }
  return data;
}
