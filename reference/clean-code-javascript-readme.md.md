# clean-code-javascript

Software engineering principles, from Robert C. Martin's book Clean Code, adapted for JavaScript. This is not a style guide. It's a guide to producing readable, reusable, and refactorable software in JavaScript.

Not every principle herein has to be strictly followed, and even fewer will be universally agreed upon. These are guidelines and nothing more, but they are ones codified over many years of collective experience by the authors of Clean Code.

Our craft of software engineering is just a bit over 50 years old, and we are still learning a lot. When software architecture is as old as architecture itself, maybe then we will have harder rules to follow. For now, let these guidelines serve as a touchstone by which to assess the quality of the JavaScript code that you and your team produce.

One more thing: knowing these won't immediately make you a better software developer, and working with them for many years doesn't mean you won't make mistakes. Every piece of code starts as a first draft, like wet clay getting shaped into its final form. Finally, we chisel away the imperfections when we review it with our peers. Don't beat yourself up for first drafts that need improvement. Beat up the code instead!

## Variables

Use meaningful and pronounceable variable names. Use the same vocabulary for the same type of variable. Use searchable names. We will read more code than we will ever write. It's important that the code we do write is readable and searchable. By not naming variables that end up being meaningful for understanding our program, we hurt our readers. Make your names searchable.

Use explanatory variables. Avoid mental mapping — explicit is better than implicit. Don't add unneeded context. If your class/object name tells you something, don't repeat that in your variable name.

Use default parameters instead of short circuiting or conditionals. Default parameters are often cleaner than short circuiting. Be aware that if you use them, your function will only provide default values for undefined arguments. Other "falsy" values such as '', "", false, null, 0, and NaN, will not be replaced by a default value.

## Functions

Function arguments (2 or fewer ideally). Limiting the amount of function parameters is incredibly important because it makes testing your function easier. Having more than three leads to a combinatorial explosion where you have to test tons of different cases with each separate argument.

Functions should do one thing. This is by far the most important rule in software engineering. When functions do more than one thing, they are harder to compose, test, and reason about. When you can isolate a function to just one action, it can be refactored easily and your code will read much cleaner. If you take nothing else away from this guide other than this, you'll be ahead of many developers.

Function names should say what they do. Functions should only be one level of abstraction. When you have more than one level of abstraction your function is usually doing too much. Splitting up functions leads to reusability and easier testing.

Remove duplicate code. Do your absolute best to avoid duplicate code. Duplicate code is bad because it means that there's more than one place to alter something if you need to change some logic.

Set default objects with Object.assign. Don't use flags as function parameters. Flags tell your user that this function does more than one thing. Functions should do one thing. Split out your functions if they are following different code paths based on a boolean.

Avoid side effects. A function produces a side effect if it does anything other than take a value in and return another value or values. A side effect could be writing to a file, modifying some global variable, or accidentally wiring all your money to a stranger.

Don't write to global functions. Polluting globals is a bad practice in JavaScript because you could clash with another library and the user of your API would be none-the-wiser until they get an exception in production.

Favor functional programming over imperative programming. JavaScript isn't a functional language in the way that Haskell is, but it has a functional flavor to it. Functional languages can be cleaner and easier to test.

Encapsulate conditionals. Avoid negative conditionals. Avoid conditionals.

Avoid type-checking (part 1). JavaScript is untyped, which means your functions can take any type of argument. Sometimes you are bitten by this freedom and it becomes tempting to do type-checking in your functions. There are many ways to avoid having to do this. The first thing to consider is consistent APIs.

Avoid type-checking (part 2). If you are working with basic primitive values like strings and integers, and you can't use polymorphism but you still feel the need to type-check, you should consider using TypeScript. It is an excellent alternative to normal JavaScript, as it provides you with static typing on top of standard JavaScript syntax.

Don't over-optimize. Modern browsers do a lot of optimization under-the-hood at runtime. A lot of times, if you are optimizing then you are just wasting your time.

Remove dead code. Dead code is just as bad as duplicate code. There's no reason to keep it in your codebase. If it's not being called, get rid of it! It will still be safe in your version history if you still need it.

## Objects and Data Structures

Use getters and setters. Using getters and setters to access data on objects could be better than simply looking for a property on an object. "Why?" you might ask. Well, here's an unorganized list of reasons why: when you want to do more beyond getting an object property, you don't have to look up and change every accessor in your codebase; makes adding validation simple when doing a set; encapsulates the internal representation; easy to add logging and error handling when getting and setting; you can lazy load your object's properties.

Make objects have private members. This can be accomplished through closures (for ES5 and below).

## Classes

Prefer ES2015/ES6 classes over ES5 plain functions. It's very difficult to get readable class inheritance, construction, and method definitions for classical ES5 classes. If you need inheritance (and be aware that you might not), then prefer ES2015/ES6 classes. However, prefer small functions over classes until you find yourself needing larger and more complex objects.

Use method chaining. This pattern is very useful in JavaScript and you see it in many libraries such as jQuery and Lodash. It allows your code to be expressive, and less verbose. For that reason, I say, use method chaining and take a look at how clean your code will be. In your class functions, simply return this at the end of every function, and you can chain further class methods onto it.

Prefer composition over inheritance. As stated famously in Design Patterns by the Gang of Four, you should prefer composition over inheritance where you can.

## SOLID

Single Responsibility Principle (SRP). As stated in Clean Code, "There should never be more than one reason for a class to change". It's tempting to jam-pack a class with a lot of functionality, like when you can only take one suitcase on your flight. The issue with this is that your class won't be conceptually cohesive and it will give it many reasons to change.

Open/Closed Principle (OCP). As stated by Bertrand Meyer, "software entities (classes, modules, functions, etc.) should be open for extension, but closed for modification." What does that mean though? This principle basically states that you should allow users to add new functionalities without changing existing code.

Liskov Substitution Principle (LSP). This is a scary term for a very simple concept. It's formally defined as "If S is a subtype of T, then objects of type T may be replaced with objects of type S (i.e., objects of type S may substitute objects of type T) without altering any of the desirable properties of that program (correctness, task performed, etc.)."

Interface Segregation Principle (ISP). JavaScript doesn't have interfaces so this principle doesn't apply as strictly as others. However, it's important and relevant even with JavaScript's lack of type system. ISP states that "Clients should not be forced to depend upon interfaces that they do not use." Interfaces are implicit contracts in JavaScript because of duck typing.

Dependency Inversion Principle (DIP). This principle states two essential things: high-level modules should not depend on low-level modules. Both should depend on abstractions; abstractions should not depend upon details. Details should depend on abstractions.

## Testing

Testing is more important than shipping. If you have no tests or an inadequate amount, then every time you ship code you won't be sure that you didn't break anything. Deciding on what constitutes an adequate amount is up to your team, but having 100% coverage (all statements and branches) is how you achieve very high confidence and developer peace of mind.

There's no excuse to not write tests. There are plenty of good JS test frameworks, so find one that your team prefers.

Single concept per test.

## Concurrency

Use Promises, not callbacks. Callbacks aren't clean, and they cause excessive amounts of nesting. With ES2015/ES6, Promises are a built-in global type. Use them!

Async/Await are even cleaner than Promises. Promises are a very clean alternative to callbacks, but ES2017/ES8 brings async and await which offer an even cleaner solution.

## Error Handling

Thrown errors are a good thing! They mean the runtime has successfully identified when something in your program has gone wrong and it's letting you know by stopping function execution on the current stack, killing the process (in Node), and notifying you in the console with a stack trace.

Don't ignore caught errors. Doing nothing with a caught error doesn't give you the ability to ever fix or react to said error. Logging the error to the console (console.log) isn't much better as often times it can get lost in a sea of things printed to the console.

Don't ignore rejected promises. For the same reason you shouldn't ignore caught errors from try/catch.

## Formatting

Formatting is subjective. Like many rules herein, there is no hard and fast rule that you must follow. The main point is DO NOT ARGUE over formatting. There are tons of tools to automate this. Use one! It's a waste of time and money for engineers to argue over formatting.

Use consistent capitalization. Function callers and callees should be close.

## Comments

Only comment things that have business logic complexity. Comments are an apology, not a requirement. Good code mostly documents itself.

Don't leave commented out code in your codebase. Version control exists for a reason. Leave old code in your history.

Don't have journal comments. Remember, use version control! There's no need for dead code, commented code, and especially journal comments. Use git log to get history!

Avoid positional markers. They usually just add noise.