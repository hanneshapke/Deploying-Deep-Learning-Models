/*
  Language: terminal console
  Author: Josh Bode <joshbode@gmail.com>
*/

var hljs = remark.highlighter.engine;

hljs.registerLanguage('terminal', function() {
  return {
    contains: [
      {
        className: 'string',
        begin: '^(([\\w.]+)@([\\w.]+)\\:(\\S+) )?\\$'
      },
      /*{
        className: 'constant',
        begin: '(:(.*) )?\\$ '
      },*/
      {
        className: 'ansi',
        begin: '<span style\\="([^"]+)">',
        end: '<\\/span>'
      }
    ]
  }
});
