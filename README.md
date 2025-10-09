# IT254-Project
### AI Shopping Cart
#### Project Description
--- 
The AI Shopping Cart is designed for a supermarket promotion where customers can fill their basket with any items they want, as long as the total stays under a set dollar limit. A camera with AI recognition identifies items such as fruits, vegetables, and meats as they are placed in or removed from the basket. The system keeps a running total of the estimated prices, and if the total goes over the limit, it plays a sound to alert the customer.

Features
--- 
- Recognizes items (fruits, vegetables, meats) with a camera and AI.
  - Requirements: *A camera module (e.g., USB webcam or Arduino camera), and a pre-trained image classification model (e.g., a HuggingFace vision model or Google Teachable Machine).*
- Tracks items added or removed from the basket.
- Calculates an estimated total based on average prices.
  - Requirements: *A price dataset (can be a simple CSV file of items and their average prices).*
- Enforces a set spending limit for the promotion.
- Plays an alert sound if the limit is exceeded.
  - Requirements: *A speaker or buzzer (Arduino-style module or system audio).*
