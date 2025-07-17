import React from "react";
import { GoogleOAuthProvider, GoogleLogin } from "@react-oauth/google";

export default function TestLogin() {
  return (
    <GoogleOAuthProvider clientId="">
      <GoogleLogin
        onSuccess={(credentialResponse) => {
          console.log("Login Success", credentialResponse);
        }}
        onError={() => {
          console.log("Login Failed");
        }}
      />
    </GoogleOAuthProvider>
  );
}
