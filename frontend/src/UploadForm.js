import React, { useState } from "react";
import axios from "axios";

function UploadForm() {
  const [imageA, setImageA] = useState(null);
  const [imageB, setImageB] = useState(null);
  const [result, setResult] = useState(null);
  const [showSimulation, setShowSimulation] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!imageA || !imageB) {
      alert("두 이미지를 모두 선택해주세요.");
      return;
    }

    const formData = new FormData();
    formData.append("imageA", imageA);
    formData.append("imageB", imageB);

    try {
      const res = await axios.post("http://localhost:8000/api/upload", formData);
      setResult(res.data);
      setShowSimulation(false);
    } catch (error) {
      console.error("업로드 실패:", error);
      alert("업로드 중 오류가 발생했습니다.");
    }
  };

  return (
    <div style={{ maxWidth: "960px", margin: "0 auto", fontFamily: "sans-serif", padding: "20px" }}>
      <h2 style={{ textAlign: "center", fontSize: "28px", marginBottom: "20px" }}>🏆 승패 예측 시스템</h2>

      <form onSubmit={handleSubmit} style={{ display: "flex", gap: "10px", marginBottom: "20px" }}>
        <div>
          <label>Player A 이미지</label><br />
          <input type="file" accept="image/*" onChange={(e) => setImageA(e.target.files[0])} />
        </div>
        <div>
          <label>Player B 이미지</label><br />
          <input type="file" accept="image/*" onChange={(e) => setImageB(e.target.files[0])} />
        </div>
        <div style={{ alignSelf: "end" }}>
          <button type="submit">업로드</button>
        </div>
      </form>

      {result && (
        <div>
          <h3>📊 예측 결과</h3>
          <div style={{ display: "flex", alignItems: "center", gap: "20px", marginBottom: "10px" }}>
            <img src={`http://localhost:8000/uploads/playerA.jpg?t=${Date.now()}`} alt="Player A" width="120" />
            <img src={`http://localhost:8000/uploads/playerB.jpg?t=${Date.now()}`} alt="Player B" width="120" />
            <p style={{ fontSize: "18px", fontWeight: "bold" }}>
              승리 확률 (A 기준): {(result.probability * 100).toFixed(2)}%
            </p>
          </div>
          <button onClick={() => setShowSimulation(true)}>🕹️ 시뮬레이션 실행</button>
        </div>
      )}

      {showSimulation && (
        <div style={{ marginTop: "30px", border: "1px solid #ccc", borderRadius: "8px", padding: "10px" }}>
          <h3>🧠 Unity 시뮬레이션</h3>
          <iframe
            src="http://localhost:8000/unity/index.html"
            width="960"
            height="600"
            style={{ border: "none", width: "100%" }}
            title="Unity Simulation"
          />
        </div>
      )}
    </div>
  );
}

export default UploadForm;
