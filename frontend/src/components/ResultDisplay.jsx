import React from "react";

const containerStyle = {
  display: "flex",
  flexWrap: "wrap",
  justifyContent: "center",
  alignItems: "center",
  gap: "20px",
  marginBottom: "20px",
};

const playerBoxStyle = {
  width: "100%",
  maxWidth: "280px",
  height: "380px",
  backgroundColor: "#f5f5f5",
  border: "2px solid #ccc",
  display: "flex",
  alignItems: "center",
  justifyContent: "center",
  overflow: "hidden",
};

const playerImageStyle = {
  width: "100%",
  height: "100%",
  objectFit: "contain",
};

const labelStyle = {
  textAlign: "center",
  fontSize: "20px",
  fontWeight: "bold",
};

function ResultDisplay({ result, setShowSimulation }) {
  return (
    <div>
      <h3
        style={{
          textAlign: "center",
          fontSize: "24px",
          marginBottom: "20px",
          color: "#2c3e50",
        }}
      >
        예측 결과
      </h3>

      <div style={containerStyle}>
        {/* Player A */}
        <div>
          <div style={playerBoxStyle}>
            <img
              src={`http://localhost:8000/uploads/playerA.jpg?t=${Date.now()}`}
              alt="Player A"
              style={playerImageStyle}
            />
          </div>
          <p style={labelStyle}>1P</p>
        </div>

        {/* VS */}
        <div style={{ fontSize: "32px", fontWeight: "bold" }}>VS</div>

        {/* Player B */}
        <div>
          <div style={playerBoxStyle}>
            <img
              src={`http://localhost:8000/uploads/playerB.jpg?t=${Date.now()}`}
              alt="Player B"
              style={playerImageStyle}
            />
          </div>
          <p style={labelStyle}>2P</p>
        </div>
      </div>

      {/* 승률 표시 */}
      <p
        style={{
          fontSize: "20px",
          fontWeight: "bold",
          textAlign: "center",
          color: "#2c3e50",
          marginTop: "10px",
          marginBottom: "20px",
        }}
      >
        승리 확률 (A 기준):{" "}
        <span style={{ color: "#e74c3c" }}>
          {(result.probability * 100).toFixed(2)}%
        </span>
      </p>

      {/* 시뮬레이션 버튼 */}
      <div style={{ textAlign: "center", marginTop: "10px" }}>
        <button
          onClick={() => setShowSimulation(true)}
          style={{
            padding: "12px 24px",
            backgroundColor: "#3498db",
            color: "#fff",
            fontSize: "16px",
            border: "none",
            borderRadius: "8px",
            cursor: "pointer",
            fontWeight: "bold",
            transition: "background-color 0.3s",
          }}
          onMouseOver={(e) => (e.target.style.backgroundColor = "#2980b9")}
          onMouseOut={(e) => (e.target.style.backgroundColor = "#3498db")}
        >
          ▶️ 시뮬레이션 실행
        </button>
      </div>
    </div>
  );
}

export default ResultDisplay;
