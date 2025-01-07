/**
 * Copyright (c) Streamlit Inc. (2018-2022) Snowflake Inc. (2022-2024)
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

// Private members use _.
/* eslint-disable no-underscore-dangle */

import { Dictionary, Field, Vector } from "apache-arrow"
import { immerable, produce } from "immer"
import range from "lodash/range"
import zip from "lodash/zip"

import { IArrow, Styler as StylerProto } from "@streamlit/lib/src/proto"
import { isNullOrUndefined } from "@streamlit/lib/src/util/utils"

import {
  Columns,
  Data,
  Index,
  parseArrowIpcBytes,
  Types,
} from "./arrowParseUtils"
import {
  DataType,
  getTypeName,
  IndexTypeName,
  RangeIndex,
  sameDataTypes,
  sameIndexTypes,
  Type,
} from "./arrowTypeUtils"

// This type should be recursive as there can be nested structures.
// Example: list[int64], list[list[unicode]], etc.
// NOTE: Commented out until we can find a way to properly define recursive types.
//
// enum DataTypeName {
//   Empty = "empty",
//   Boolean = "bool",
//   Number = "int64",
//   Float = "float64",
//   String = "unicode",
//   Date = "date", // "datetime", "datetimetz"
//   Bytes = "bytes",
//   Object = "object",
//   List = "list[int64]",
// }

/** DataFrame's Styler information. */
interface Styler {
  /** Styler's UUID. */
  uuid: string

  /** Optional user-specified caption. */
  caption: string | null

  /** DataFrame's CSS styles. */
  styles: string | null

  /**
   * Stringified versions of each cell in the DataFrame, in the
   * user-specified format.
   */
  displayValues: Quiver
}

/** Dimensions of the DataFrame. */
interface DataFrameDimensions {
  headerRows: number
  headerColumns: number
  dataRows: number
  dataColumns: number
  rows: number
  columns: number
}

/**
 * There are 4 cell types:
 *  - blank, cells that are not part of index headers, column headers, or data
 *  - index, index header cells
 *  - columns, column header cells
 *  - data, data cells
 */
export enum DataFrameCellType {
  BLANK = "blank",
  INDEX = "index",
  COLUMNS = "columns",
  DATA = "data",
}

/** Data for a single cell in a DataFrame. */
export interface DataFrameCell {
  /** The cell's type (blank, index, columns, or data). */
  type: DataFrameCellType

  /** The cell's CSS id, if the DataFrame has Styler. */
  cssId?: string

  /** The cell's CSS class. */
  cssClass: string

  /** The cell's content. */
  content: DataType

  /** The cell's content type. */
  // For "blank" cells "contentType" is undefined.
  contentType?: Type

  /** The cell's field. */
  field?: Field

  /**
   * The cell's formatted content string, if the DataFrame was created with a Styler.
   * If the DataFrame is unstyled, displayContent will be undefined, and display
   * code should apply a default formatting to the `content` value instead.
   */
  displayContent?: string
}

/**
 * Parses data from an Arrow table, and stores it in a row-major format
 * (which is more useful for our frontend display code than Arrow's columnar format).
 */
export class Quiver {
  /**
   * Plain objects (objects without a prototype), arrays, Maps and Sets are always drafted by Immer.
   * Every other object must use the immerable symbol to mark itself as compatible with Immer.
   * When one of these objects is mutated within a producer, its prototype is preserved between copies.
   * Source: https://immerjs.github.io/immer/complex-objects/
   */
  [immerable] = true

  /** DataFrame's index (matrix of row names). */
  private _index: Index

  /** DataFrame's column labels (matrix of column names). */
  private _columns: Columns

  /** DataFrame's index names. */
  private _indexNames: string[]

  /** DataFrame's data. */
  private _data: Data

  /** Definition for DataFrame's fields. */
  private _fields: Record<string, Field<any>>

  /** Types for DataFrame's index and data. */
  private _types: Types

  /** [optional] DataFrame's Styler data. This will be defined if the user styled the dataframe. */
  private readonly _styler?: Styler

  constructor(element: IArrow) {
    const { index, columns, data, types, fields, indexNames } =
      parseArrowIpcBytes(element.data)

    const styler = element.styler
      ? parseStyler(element.styler as StylerProto)
      : undefined

    // The assignment is done below to avoid partially populating the instance
    // if an error is thrown.
    this._index = index
    this._columns = columns
    this._data = data
    this._types = types
    this._fields = fields
    this._styler = styler
    this._indexNames = indexNames
  }

  /**
   * Returns the categorical options defined for a given data column.
   * Returns undefined if the column is not categorical.
   *
   * This function only works for non-index columns and expects the index at 0
   * for the first non-index data column.
   */
  public getCategoricalOptions(dataColumnIndex: number): string[] | undefined {
    const { dataColumns: numDataColumns } = this.dimensions

    if (dataColumnIndex < 0 || dataColumnIndex >= numDataColumns) {
      throw new Error(`Column index is out of range: ${dataColumnIndex}`)
    }

    if (!(this._fields[String(dataColumnIndex)].type instanceof Dictionary)) {
      // This is not a categorical column
      return undefined
    }

    const categoricalDict =
      this._data.getChildAt(dataColumnIndex)?.data[0]?.dictionary
    if (categoricalDict) {
      // get all values into a list
      const values = []

      for (let i = 0; i < categoricalDict.length; i++) {
        values.push(categoricalDict.get(i))
      }
      return values
    }
    return undefined
  }

  /** Concatenate the original DataFrame index with the given one. */
  private concatIndexes(otherIndex: Index, otherIndexTypes: Type[]): Index {
    // If one of the `index` arrays is empty, return the other one.
    // Otherwise, they will have different types and an error will be thrown.
    if (otherIndex.length === 0) {
      return this._index
    }
    if (this._index.length === 0) {
      return otherIndex
    }

    // Make sure indexes have same types.
    if (!sameIndexTypes(this._types.index, otherIndexTypes)) {
      const receivedIndexTypes = otherIndexTypes.map(index =>
        getTypeName(index)
      )
      const expectedIndexTypes = this._types.index.map(index =>
        getTypeName(index)
      )

      throw new Error(`
Unsupported operation. The data passed into \`add_rows()\` must have the same
index signature as the original data.

In this case, \`add_rows()\` received \`${JSON.stringify(receivedIndexTypes)}\`
but was expecting \`${JSON.stringify(expectedIndexTypes)}\`.
`)
    }

    if (this._types.index.length === 0) {
      // This should never happen!
      throw new Error("There was an error while parsing index types.")
    }

    // NOTE: "range" index cannot be a part of a multi-index, i.e.
    // if the index type is "range", there will only be one element in the index array.
    if (this._types.index[0].pandas_type === IndexTypeName.RangeIndex) {
      // Continue the sequence for a "range" index.
      // NOTE: The metadata of the original index will be used, i.e.
      // if both indexes are of type "range" and they have different
      // metadata (start, step, stop) values, the metadata of the given
      // index will be ignored.
      const { step, stop } = this._types.index[0].meta as RangeIndex
      otherIndex = [
        range(
          stop,
          // End is not inclusive
          stop + otherIndex[0].length * step,
          step
        ),
      ]
    }

    // Concatenate each index with its counterpart in the other table
    const zipped = zip(this._index, otherIndex)
    // @ts-expect-error We know the two indexes are of the same size
    return zipped.map(a => a[0].concat(a[1]))
  }

  /** Concatenate the original DataFrame data with the given one. */
  private concatData(otherData: Data, otherDataType: Type[]): Data {
    // If one of the `data` arrays is empty, return the other one.
    // Otherwise, they will have different types and an error will be thrown.
    if (otherData.numCols === 0) {
      return this._data
    }
    if (this._data.numCols === 0) {
      return otherData
    }

    // Make sure `data` arrays have the same types.
    if (!sameDataTypes(this._types.data, otherDataType)) {
      const receivedDataTypes = otherDataType.map(t => t.pandas_type)
      const expectedDataTypes = this._types.data.map(t => t.pandas_type)

      throw new Error(`
Unsupported operation. The data passed into \`add_rows()\` must have the same
data signature as the original data.

In this case, \`add_rows()\` received \`${JSON.stringify(receivedDataTypes)}\`
but was expecting \`${JSON.stringify(expectedDataTypes)}\`.
`)
    }

    // Remove extra columns from the "other" DataFrame.
    // Columns from otherData are used by index without checking column names.
    const slicedOtherData = otherData.selectAt(range(0, this._data.numCols))
    return this._data.concat(slicedOtherData)
  }

  /** Concatenate index and data types. */
  private concatTypes(otherTypes: Types): Types {
    const index = this.concatIndexTypes(otherTypes.index)
    const data = this.concatDataTypes(otherTypes.data)
    return { index, data }
  }

  /** Concatenate index types. */
  private concatIndexTypes(otherIndexTypes: Type[]): Type[] {
    // If one of the `types` arrays is empty, return the other one.
    // Otherwise, an empty array will be returned.
    if (otherIndexTypes.length === 0) {
      return this._types.index
    }
    if (this._types.index.length === 0) {
      return otherIndexTypes
    }

    // Make sure indexes have same types.
    if (!sameIndexTypes(this._types.index, otherIndexTypes)) {
      const receivedIndexTypes = otherIndexTypes.map(index =>
        getTypeName(index)
      )
      const expectedIndexTypes = this._types.index.map(index =>
        getTypeName(index)
      )

      throw new Error(`
Unsupported operation. The data passed into \`add_rows()\` must have the same
index signature as the original data.

In this case, \`add_rows()\` received \`${JSON.stringify(receivedIndexTypes)}\`
but was expecting \`${JSON.stringify(expectedIndexTypes)}\`.
`)
    }

    // TL;DR This sets the new stop value.
    return this._types.index.map(indexType => {
      // NOTE: "range" index cannot be a part of a multi-index, i.e.
      // if the index type is "range", there will only be one element in the index array.
      if (indexType.pandas_type === IndexTypeName.RangeIndex) {
        const { stop, step } = indexType.meta as RangeIndex
        const {
          start: otherStart,
          stop: otherStop,
          step: otherStep,
        } = otherIndexTypes[0].meta as RangeIndex
        const otherRangeIndexLength = (otherStop - otherStart) / otherStep
        const newStop = stop + otherRangeIndexLength * step
        return {
          ...indexType,
          meta: {
            ...indexType.meta,
            stop: newStop,
          },
        }
      }
      return indexType
    })
  }

  /** Concatenate types of data columns. */
  private concatDataTypes(otherDataTypes: Type[]): Type[] {
    if (this._types.data.length === 0) {
      return otherDataTypes
    }

    return this._types.data
  }

  /** DataFrame's index (matrix of row names). */
  public get index(): Index {
    return this._index
  }

  /** DataFrame's index names. */
  public get indexNames(): string[] {
    return this._indexNames
  }

  /** DataFrame's column labels (matrix of column names). */
  public get columns(): Columns {
    return this._columns
  }

  /** DataFrame's data. */
  public get data(): Data {
    return this._data
  }

  /** Types for DataFrame's index and data. */
  public get types(): Types {
    return this._types
  }

  /**
   * The DataFrame's CSS id, if it has one.
   *
   * If the DataFrame has a Styler, the  CSS id is `T_${StylerUUID}`. Otherwise,
   * it's undefined.
   *
   * This id is used by styled tables and styled dataframes to associate
   * the Styler CSS with the styled data.
   */
  public get cssId(): string | undefined {
    if (
      isNullOrUndefined(this._styler) ||
      isNullOrUndefined(this._styler.uuid)
    ) {
      return undefined
    }

    return `T_${this._styler.uuid}`
  }

  /** The DataFrame's CSS styles, if it has a Styler. */
  public get cssStyles(): string | undefined {
    return this._styler?.styles || undefined
  }

  /** The DataFrame's caption, if it's been set. */
  public get caption(): string | undefined {
    return this._styler?.caption || undefined
  }

  /** The DataFrame's dimensions. */
  public get dimensions(): DataFrameDimensions {
    const headerColumns = this._index.length || this.types.index.length || 1
    const headerRows = this._columns.length || 1
    const dataRows = this._data.numRows || 0
    const dataColumns = this._data.numCols || this._columns?.[0]?.length || 0

    const rows = headerRows + dataRows
    const columns = headerColumns + dataColumns

    return {
      headerRows,
      headerColumns,
      dataRows,
      dataColumns,
      rows,
      columns,
    }
  }

  /** True if the DataFrame has no index, columns, and data. */
  public isEmpty(): boolean {
    return (
      this._index.length === 0 &&
      this._columns.length === 0 &&
      this._data.numRows === 0 &&
      this._data.numCols === 0
    )
  }

  /** Return a single cell in the table. */
  public getCell(rowIndex: number, columnIndex: number): DataFrameCell {
    const { headerRows, headerColumns, rows, columns } = this.dimensions

    if (rowIndex < 0 || rowIndex >= rows) {
      throw new Error(`Row index is out of range: ${rowIndex}`)
    }
    if (columnIndex < 0 || columnIndex >= columns) {
      throw new Error(`Column index is out of range: ${columnIndex}`)
    }

    const isBlankCell = rowIndex < headerRows && columnIndex < headerColumns
    const isIndexCell = rowIndex >= headerRows && columnIndex < headerColumns
    const isColumnsCell = rowIndex < headerRows && columnIndex >= headerColumns

    if (isBlankCell) {
      // Blank cells include `blank`.
      const cssClass = ["blank"]
      if (columnIndex > 0) {
        cssClass.push(`level${rowIndex}`)
      }

      return {
        type: DataFrameCellType.BLANK,
        cssClass: cssClass.join(" "),
        content: "",
      }
    }

    if (isIndexCell) {
      const dataRowIndex = rowIndex - headerRows

      const cssId = this._styler?.uuid
        ? `${this.cssId}level${columnIndex}_row${dataRowIndex}`
        : undefined

      // Index label cells include:
      // - row_heading
      // - row<n> where n is the numeric position of the row
      // - level<k> where k is the level in a MultiIndex
      const cssClass = [
        `row_heading`,
        `level${columnIndex}`,
        `row${dataRowIndex}`,
      ].join(" ")

      const contentType = this._types.index[columnIndex]
      const content = this.getIndexValue(dataRowIndex, columnIndex)
      let field = this._fields[`__index_level_${String(columnIndex)}__`]
      if (field === undefined) {
        // If the index column has a name, we need to get it differently:
        field = this._fields[String(columns - headerColumns)]
      }
      return {
        type: DataFrameCellType.INDEX,
        cssId,
        cssClass,
        content,
        contentType,
        field,
      }
    }

    if (isColumnsCell) {
      const dataColumnIndex = columnIndex - headerColumns

      // Column label cells include:
      // - col_heading
      // - col<n> where n is the numeric position of the column
      // - level<k> where k is the level in a MultiIndex
      const cssClass = [
        `col_heading`,
        `level${rowIndex}`,
        `col${dataColumnIndex}`,
      ].join(" ")

      return {
        type: DataFrameCellType.COLUMNS,
        cssClass,
        content: this._columns[rowIndex][dataColumnIndex],
        // ArrowJS automatically converts "columns" cells to strings.
        // Keep ArrowJS structure for consistency.
        contentType: {
          pandas_type: IndexTypeName.UnicodeIndex,
          numpy_type: "object",
        },
      }
    }

    const dataRowIndex = rowIndex - headerRows
    const dataColumnIndex = columnIndex - headerColumns

    const cssId = this._styler?.uuid
      ? `${this.cssId}row${dataRowIndex}_col${dataColumnIndex}`
      : undefined

    // Data cells include `data`.
    const cssClass = [
      "data",
      `row${dataRowIndex}`,
      `col${dataColumnIndex}`,
    ].join(" ")

    const contentType = this._types.data[dataColumnIndex]
    const field = this._fields[String(dataColumnIndex)]
    const content = this.getDataValue(dataRowIndex, dataColumnIndex)
    const displayContent = this._styler?.displayValues
      ? (this._styler.displayValues.getCell(rowIndex, columnIndex)
          .content as string)
      : undefined

    return {
      type: DataFrameCellType.DATA,
      cssId,
      cssClass,
      content,
      contentType,
      displayContent,
      field,
    }
  }

  public getIndexValue(rowIndex: number, columnIndex: number): any {
    const index = this._index[columnIndex]
    const value =
      index instanceof Vector ? index.get(rowIndex) : index[rowIndex]
    return value
  }

  public getDataValue(rowIndex: number, columnIndex: number): any {
    return this._data.getChildAt(columnIndex)?.get(rowIndex)
  }

  /**
   * Add the contents of another table (data + indexes) to this table.
   * Extra columns will not be created.
   */
  public addRows(other: Quiver): Quiver {
    if (this._styler || other._styler) {
      throw new Error(`
Unsupported operation. \`add_rows()\` does not support Pandas Styler objects.

If you do not need the Styler's styles, try passing the \`.data\` attribute of
the Styler object instead to concatenate just the underlying dataframe.

For example:
\`\`\`
st.add_rows(my_styler.data)
\`\`\`
`)
    }

    // Don't do anything if the incoming DataFrame is empty.
    if (other.isEmpty()) {
      return produce(this, (draft: Quiver) => draft)
    }

    // We need to handle this separately, as columns need to be reassigned.
    // We don't concatenate columns in the general case.
    if (this.isEmpty()) {
      return produce(other, (draft: Quiver) => draft)
    }

    // Concatenate all data into temporary variables. If any of
    // these operations fail, an error will be thrown and we'll prematurely
    // exit the function.
    const index = this.concatIndexes(other._index, other._types.index)
    const data = this.concatData(other._data, other._types.data)
    const types = this.concatTypes(other._types)

    // If we get here, then we had no concatenation errors.
    return produce(this, (draft: Quiver) => {
      draft._index = index
      draft._data = data
      draft._types = types
    })
  }
}

/** Parse styler information from proto. */
function parseStyler(styler: StylerProto): Styler {
  return {
    uuid: styler.uuid,
    caption: styler.caption,
    styles: styler.styles,

    // Recursively create a new Quiver instance for Styler's display values.
    // This values will be used for rendering the DataFrame, while the original values
    // will be used for sorting, etc.
    displayValues: new Quiver({ data: styler.displayValues }),
  }
}
