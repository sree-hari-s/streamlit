/**
 * Copyright (c) Streamlit Inc. (2018-2022) Snowflake Inc. (2022-2025)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

const { RuleTester } = require("eslint")
const useStrictNullEqualityChecks = require("./use-strict-null-equality-checks")

const ruleTester = new RuleTester({})

// Throws error if the tests do not pass
ruleTester.run(
  "use-strict-null-equality-checks",
  useStrictNullEqualityChecks,
  {
    valid: [
      {
        code: "isNullOrUndefined(foo)",
      },
      {
        code: "notNullOrUndefined(foo)",
      },
    ],
    invalid: [
      {
        code: "foo == null",
        output: "isNullOrUndefined(foo)",
        errors: 1,
      },
      {
        code: "foo != null",
        output: "notNullOrUndefined(foo)",
        errors: 1,
      },
      {
        code: "foo == undefined",
        output: "isNullOrUndefined(foo)",
        errors: 1,
      },
      {
        code: "foo != undefined",
        output: "notNullOrUndefined(foo)",
        errors: 1,
      },
    ],
  }
)

console.log("All tests passed!")
