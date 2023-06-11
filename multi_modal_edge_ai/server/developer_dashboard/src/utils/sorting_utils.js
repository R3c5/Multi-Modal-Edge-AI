class SortingUtils {

    /**
     * Sorts the data array alphabetically based on the specified column and direction.
     * @param {Array} data - The array of data to be sorted.
     * @param {string} column - The column to sort by.
     * @param {string} direction - The sorting direction ('asc' for ascending, 'desc' for descending).
     * @returns {Array} The sorted array.
     */
    static sortAlphabetically(data, column, direction) {
        return [...data].sort((a, b) =>
            direction === 'asc'
                ? a[column].localeCompare(b[column])
                : b[column].localeCompare(a[column])
        );
    }
    static sortNumerically(data, column, direction) {
        return [...data].sort((a, b) =>
            direction === 'asc' ? a[column] - b[column] : b[column] - a[column]);
    }

    /**
     * Sorts the data array by date and time based on the specified column and direction.
     * @param {Array} data - The array of data to be sorted.
     * @param {string} column - The column to sort by.
     * @param {string} direction - The sorting direction ('asc' for ascending, 'desc' for descending).
     * @returns {Array} The sorted array.
     */
    static sortByDateTime(data, column, direction) {
        return [...data].sort((a, b) => {
            const dateA = new Date(a[column].replace("\n", "T"));
            const dateB = new Date(b[column].replace("\n", "T"));
            return direction === 'asc' ? dateA - dateB : dateB - dateA;
        });
    }

    /**
     * Sorts the data array by IP address based on the specified column and direction.
     * @param {Array} data - The array of data to be sorted.
     * @param {string} column - The column to sort by.
     * @param {string} direction - The sorting direction ('asc' for ascending, 'desc' for descending).
     * @returns {Array} The sorted array.
     */
    static sortByIPAddress(data, column, direction) {
        return [...data].sort((a, b) => {
            const ipA = a[column].split('.').map(Number);
            const ipB = b[column].split('.').map(Number);

            for (let i = 0; i < ipA.length; i++) {
                if (ipA[i] < ipB[i]) {
                    return direction === 'asc' ? -1 : 1;
                }
                if (ipA[i] > ipB[i]) {
                    return direction === 'asc' ? 1 : -1;
                }
            }

            return 0;
        });
    }

}

export default SortingUtils;