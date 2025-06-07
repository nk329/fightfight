// import React, { useState } from "react";
// import axios from "axios";

// function App() {
//   const [imageA, setImageA] = useState(null);
//   const [imageB, setImageB] = useState(null);
//   const [result, setResult] = useState(null);

//   const handleUpload = async () => {
//     if (!imageA || !imageB) {
//       alert("두 이미지를 모두 선택해주세요.");
//       return;
//     }

//     const formData = new FormData();
//     formData.append("imageA", imageA);
//     formData.append("imageB", imageB);

//     try {
//       const res = await axios.post("http://127.0.0.1:8000/api/upload", formData, {
//         headers: { "Content-Type": "multipart/form-data" },
//       });

//       console.log("✅ 서버 응답:", res.data);

//       const { faceA, faceB, probability } = res.data;

//       if (!faceA || !faceB || probability === undefined) {
//         alert("⚠️ 서버 응답이 예상과 다릅니다.");
//         return;
//       }

//       setResult({ faceA, faceB, probability });
//     } catch (err) {
//       console.error("❌ 업로드 중 오류:", err);
//       alert("업로드 중 오류가 발생했습니다.");
//     }
//   };

//   return (
//     <div style={{ padding: "20px", fontFamily: "sans-serif" }}>
//       <h1>👊 승부 예측기</h1>

//       <div style={{ marginBottom: "10px" }}>
//         <input type="file" accept="image/*" onChange={(e) => setImageA(e.target.files[0])} />
//         <input type="file" accept="image/*" onChange={(e) => setImageB(e.target.files[0])} />
//         <button onClick={handleUpload}>업로드 및 예측</button>
//       </div>

//       {result && (
//         <div style={{ marginTop: "20px" }}>
//           <h2>🎯 예측 결과</h2>
//           <div style={{ display: "flex", gap: "20px" }}>
//             <div>
//               <img
//                 src={`http://127.0.0.1:8000${result.faceA}?t=${Date.now()}`}
//                 alt="Face A"
//                 width="150"
//               />
//               <p>Player A</p>
//             </div>
//             <div>
//               <img
//                 src={`http://127.0.0.1:8000${result.faceB}?t=${Date.now()}`}
//                 alt="Face B"
//                 width="150"
//               />
//               <p>Player B</p>
//             </div>
//           </div>
//           <p style={{ marginTop: "10px", fontSize: "18px" }}>
//             🏆 <strong>Player A 승리 확률:</strong> {(result.probability * 100).toFixed(2)}%
//           </p>
//         </div>
//       )}
//     </div>
//   );
// }

// export default App;
import React from "react";
import UploadForm from "./UploadForm"; // 경로 확인

function App() {
  return (
    <div style={{ padding: "20px", fontFamily: "sans-serif" }}>
      <UploadForm />
    </div>
  );
}

export default App;
