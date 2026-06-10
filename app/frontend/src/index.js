import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";

// 使用 createRoot 方法创建根节点
const root = ReactDOM.createRoot(document.getElementById("root"));

// 渲染应用
root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);