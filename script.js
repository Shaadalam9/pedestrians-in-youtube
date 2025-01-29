const { generate } = require('./youtube-po-token-generator')

generate().then(console.log, console.error)