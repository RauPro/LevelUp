# Use Node.js 18 LTS
FROM node:22-alpine

# Set working directory
WORKDIR /app

# Copy package.json and package-lock.json (if available)
COPY package*.json ./

# Install dependencies
RUN npm ci

# Copy the rest of the application code
COPY . .

# Expose port 4200 (Angular default)
EXPOSE 4200

# Start the Angular development server
CMD ["npm", "start", "--", "--host", "0.0.0.0", "--port", "4200"]
