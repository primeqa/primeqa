<!---
Copyright 2022 PrimeQA Team

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

<div align="center">
    <img src="_static/img/PrimeQA.png" width="150"/>
</div>

# Playground Tooling        

<br>

[![LICENSE|Apache2.0](https://img.shields.io/github/license/saltstack/salt?color=blue)](https://www.apache.org/licenses/LICENSE-2.0.txt)

<h2>✔️ Getting Started</h2> 

- [Repository](https://github.ibm.com/IBM-Research-AI/playground-ui)        
- [Demo](http://mnlp-qa-dev-2.sl.cloud9.ibm.com:82/)    
- This project was bootstrapped with [Create React App](https://github.com/facebook/create-react-app)   

<h2>✅ Prerequisites</h2>

- [Yarn](https://classic.yarnpkg.com/en/docs/install)

<h2>🧩 Setup Local Environment</h2>

- Dowload all necessary packages to build and deploy the application: `yarn install`        
- Open `.env` files and set `REACT_APP_API_URL` to point to playground-services:        
    - dev-2 machine: http://9.59.197.117:{PORT}      
    - local: http://0.0.0.0:{PORT}      

<h2>💻 Run Locally</h2>

- Run the app in the _*development mode*_: `yarn start`        
- Open [http://localhost:8888](http://localhost:8888) to view it in the browser.               

<h2>💻 Setup & Run Docker</h2>

This allows us to run the build in a node image and server the app using an nginx image.        
The final Docker image will just contain the build folder and nothing else      
(the project files were only used by to build the project in the builder layer, which then gets thrown away)      
it's just an intermmediary step.        

- files: *Dockerfile*, *nginx.conf*, *.dockerignore*      
- `docker build . -t primeqa_ui`       
- `docker run --rm --name primeqa_ui -d -p 82:82 primeqa_ui:$(cat VERSION)`              
    - 82 -> public port to access     
    - 82 -> container expose port  
- stop container: `docker stop  primeqa_ui`        
- remove container: `docker rm primeqa_ui`     
- remove image: `docker rmi primeqa_ui`        