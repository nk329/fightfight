import React, { useState } from "react";
import axios from "axios";

function UploadForm() {
  const [imageA, setImageA] = useState(null);
  const [imageB, setImageB] = useState(null);
  const [result, setResult] = useState(null);

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
    } catch (error) {
      console.error("업로드 실패:", error);
      alert("업로드 중 오류가 발생했습니다.");
    }
  };

  return (
    <div>
      <h2>🏋️ 승패 예측 이미지 업로드</h2>
      <form onSubmit={handleSubmit}>
        <input type="file" accept="image/*" onChange={(e) => setImageA(e.target.files[0])} />
        <input type="file" accept="image/*" onChange={(e) => setImageB(e.target.files[0])} />
        <button type="submit">업로드</button>
      </form>

      {result && (
        <div>
          <h3>🎯 예측 결과</h3>
         <img src={`http://localhost:8000${result.faceA}?t=${Date.now()}`} alt="Face A" width="100" />
         <img src={`http://localhost:8000${result.faceB}?t=${Date.now()}`} alt="Face B" width="100" />
          <p>승리 확률 (A 기준): {result.probability * 100}%</p>
        </div>
      )}
    </div>
  );
}

export default UploadForm;
