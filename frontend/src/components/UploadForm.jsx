import React, { useState } from "react";
import axios from "axios";
import ResultDisplay from "./ResultDisplay";

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
    <div style={{
      maxWidth: "960px",
      margin: "0 auto",
      fontFamily: "sans-serif",
      padding: "20px",
      backgroundColor: "#ffffff",
      borderRadius: "12px",
      boxShadow: "0 4px 12px rgba(0, 0, 0, 0.1)"
    }}>
      <h2 style={{
        textAlign: "center",
        fontSize: "28px",
        marginBottom: "20px",
        color: "#333"
      }}>🏆 승패 예측 시스템</h2>

      {/* 업로드 폼 */}
      <form onSubmit={handleSubmit} style={{
        display: "flex",
        flexWrap: "wrap",
        gap: "20px",
        justifyContent: "space-between",
        marginBottom: "20px"
      }}>
        <div style={{ flex: "1 1 200px", minWidth: "200px" }}>
          <label style={{ display: "block", marginBottom: "8px", fontWeight: "bold" }}>Player A 이미지</label>
          <input
            type="file"
            accept="image/*"
            onChange={(e) => setImageA(e.target.files[0])}
            style={{
              width: "100%",
              padding: "8px",
              borderRadius: "6px",
              border: "1px solid #ccc",
              cursor: "pointer"
            }}
          />
        </div>

        <div style={{ flex: "1 1 200px", minWidth: "200px" }}>
          <label style={{ display: "block", marginBottom: "8px", fontWeight: "bold" }}>Player B 이미지</label>
          <input
            type="file"
            accept="image/*"
            onChange={(e) => setImageB(e.target.files[0])}
            style={{
              width: "100%",
              padding: "8px",
              borderRadius: "6px",
              border: "1px solid #ccc",
              cursor: "pointer"
            }}
          />
        </div>

        <div style={{ alignSelf: "end", flex: "0 1 auto" }}>
          <button type="submit" style={{
            padding: "10px 20px",
            backgroundColor: "#4CAF50",
            color: "#fff",
            border: "none",
            borderRadius: "6px",
            cursor: "pointer",
            fontWeight: "bold"
          }}>
            업로드
          </button>
        </div>
      </form>

      {/* 결과 화면 */}
      {result && (
        <ResultDisplay result={result} setShowSimulation={setShowSimulation} />
      )}

      {/* Unity 시뮬레이션 */}
      {showSimulation && (
        <div style={{
          marginTop: "30px",
          border: "1px solid #ccc",
          borderRadius: "8px",
          padding: "10px"
        }}>
          <h3>Unity 시뮬레이션</h3>
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
