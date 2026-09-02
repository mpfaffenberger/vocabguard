# Create React App

Create React apps with no build configuration.

Create React App works on macOS, Windows, and Linux. If something doesn't work, please file an issue. If you have questions or need help, please ask in GitHub Discussions.

## Quick Overview

npx create-react-app my-app
cd my-app
npm start

If you've previously installed create-react-app globally via npm install -g create-react-app, we recommend you uninstall the package using npm uninstall -g create-react-app or yarn global remove create-react-app to ensure that npx always uses the latest version.

Then open http://localhost:3000/ to see your app. When you're ready to deploy to production, create a minified bundle with npm run build.

## Creating an App

You'll need to have Node 14.0.0 or later version on your local development machine (but it's not required on the server). We recommend using the latest LTS version. You can use nvm (macOS/Linux) or nvm-windows to switch Node versions between different projects.

To create a new app, you may choose one of the following methods:

- npx: npx create-react-app my-app
- npm: npm init react-app my-app
- Yarn: yarn create react-app my-app

It will create a directory called my-app inside the current folder. Inside that directory, it will generate the initial project structure and install the transitive dependencies.

No configuration or complicated folder structures, only the files you need to build your app. Once the installation is done, you can open your project folder.

## Available Commands

npm start or yarn start: Runs the app in development mode. Open http://localhost:3000 to view it in the browser. The page will automatically reload if you make changes to the code. You will see the build errors and lint warnings in the console.

npm test or yarn test: Runs the test watcher in an interactive mode. By default, runs tests related to files changed since the last commit.

npm run build or yarn build: Builds the app for production to the build folder. It correctly bundles React in production mode and optimizes the build for the best performance. The build is minified and the filenames include the hashes. Your app is ready to be deployed.

## User Guide

You can find detailed instructions on using Create React App and many tips in its documentation.

## Philosophy

- One Dependency: There is only one build dependency. It uses webpack, Babel, ESLint, and other amazing projects, but provides a cohesive curated experience on top of them.

- No Configuration Required: You don't need to configure anything. A reasonably good configuration of both development and production builds is handled for you so you can focus on writing code.

- No Lock-In: You can "eject" to a custom setup at any time. Run a single command, and all the configuration and build dependencies will be moved directly into your project, so you can pick up right where you left off.

## What's Included

Your environment will have everything you need to build a modern single-page React app:

- React, JSX, ES6, TypeScript and Flow syntax support.
- Language extras beyond ES6 like the object spread operator.
- Autoprefixed CSS, so you don't need -webkit- or other prefixes.
- A fast interactive unit test runner with built-in support for coverage reporting.
- A live development server that warns about common mistakes.
- A build script to bundle JS, CSS, and images for production, with hashes and sourcemaps.
- An offline-first service worker and a web app manifest, meeting all the Progressive Web App criteria.
- Hassle-free updates for the above tools with a single dependency.

The tradeoff is that these tools are preconfigured to work in a specific way. If your project needs more customization, you can "eject" and customize it, but then you will need to maintain this configuration.

## Popular Alternatives

Create React App is a great fit for:

- Learning React in a comfortable and feature-rich development environment.
- Starting new single-page React applications.
- Creating examples with React for your libraries and components.

Here are a few common cases where you might want to try something else:

- If you want to try React without hundreds of transitive build tool dependencies, consider using a single HTML file or an online sandbox instead.

- If you need to integrate React code with a server-side template framework like Rails, Django or Symfony, or if you're not building a single-page app, consider using nwb or Neutrino which are more flexible.

- If you need to publish a React component, nwb can also do this.

- If you want to do server rendering with React and Node.js, check out Next.js or Razzle. Create React App is agnostic of the backend, and only produces static HTML/JS/CSS bundles.

- If your website is mostly static (for example, a portfolio or a blog), consider using Gatsby or Next.js. Unlike Create React App, Gatsby pre-renders the website into HTML at build time.

- Finally, if you need more customization, check out Neutrino and its React preset.

## React Native

Looking for something similar, but for React Native? Check out Expo CLI.

## Contributing

We'd love to have your helping hand on create-react-app! See CONTRIBUTING.md for more information on what we're looking for and how to get started.

## Supporting Create React App

Create React App is a community maintained project and all contributors are volunteers. If you'd like to support the future development of Create React App then please consider donating to our Open Collective.

## Credits

This project exists thanks to all the people who contribute. We are grateful to the authors of existing related projects for their ideas and collaboration.

## License

Create React App is open source software licensed as MIT.