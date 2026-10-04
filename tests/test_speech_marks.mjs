import assert from 'node:assert/strict';
import test from 'node:test';
import { readFileSync } from 'node:fs';
import ts from 'typescript';

const source = readFileSync(new URL('../src/audio/speechMarks.ts', import.meta.url), 'utf8');
const { outputText } = ts.transpileModule(source, {
  compilerOptions: { module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2020 },
});
const { chunkForPosition, markAtTime, timeAtPosition, resolvePlaybackPosition } = await import(
  `data:text/javascript;base64,${Buffer.from(outputText).toString('base64')}`
);

const marks = [
  { timeMs: 100, docCharStart: 50, docCharEnd: 55 },
  { timeMs: 700, docCharStart: 57, docCharEnd: 62 },
];

test('time and character mapping uses preceding word, including boundaries', () => {
  assert.equal(timeAtPosition(marks, 49), 0);
  assert.equal(timeAtPosition(marks, 50), 100);
  assert.equal(timeAtPosition(marks, 56), 100);
  assert.equal(timeAtPosition(marks, 57), 700);
  assert.equal(timeAtPosition(marks, 100), 700);
  assert.equal(markAtTime(marks, 0), marks[0]);
  assert.equal(markAtTime(marks, 699), marks[0]);
  assert.equal(markAtTime(marks, 700), marks[1]);
  assert.equal(markAtTime(marks, 10000), marks[1]);
  assert.equal(timeAtPosition([], 100), 0);
  assert.equal(markAtTime([], 100), undefined);
});

test('character resume targets select the containing chunk', () => {
  const chunks = [
    { doc_char_start: 0, doc_char_end: 50 },
    { doc_char_start: 50, doc_char_end: 100 },
  ];
  assert.equal(chunkForPosition(chunks, 49), 0);
  assert.equal(chunkForPosition(chunks, 50), 1);
  assert.equal(chunkForPosition(chunks, 100), 1);
  assert.equal(chunkForPosition([], 0), 0);
});


test('instruction positions wait for marks and discard local audio coordinates', async () => {
  let release;
  const pending = new Promise((resolve) => { release = resolve; });
  const point = { documentId: 'revision', docCharPosition: 50, chunkId: 1, offsetMs: 900 };
  let ready = false;
  const result = resolvePlaybackPosition(point, [{doc_char_start: 0}, {doc_char_start: 50}],
    () => pending).then((value) => { ready = true; return value; });
  await Promise.resolve();
  assert.equal(ready, false);
  release([{timeMs: 700, docCharStart: 57, docCharEnd: 62}]);
  assert.deepEqual(await result, {documentId: 'revision', docCharPosition: 57});
  await assert.rejects(resolvePlaybackPosition(point, [{doc_char_start: 0}, {doc_char_start: 50}],
    async () => { throw new Error('alignment failed'); }), /alignment failed/);
});

test('empty slots keep their indices when selecting chunks', () => {
  const chunks = [{doc_char_end: 0}, {doc_char_end: 5}, {doc_char_end: 5}, {doc_char_end: 9}];
  assert.equal(chunkForPosition(chunks, 0), 1);
  assert.equal(chunkForPosition(chunks, 5), 3);
  assert.equal(chunkForPosition(chunks, 9), 3);
});
