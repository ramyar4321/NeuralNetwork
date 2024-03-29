#ifndef MATRIX_TESTING
#define MATRIX_TESTING


namespace gpu{
    /**
     * MatrixTest class that will test important methodes of 
     * the Matrix class. Tests will focus on success of operations
     * on d_mat. The testing will not be rigorous.
    */
    class MatrixTesting{

        public:

            MatrixTesting();

            void testCopyConstructor();
            void testVectorConstructor();
            void testDeepCopy();
            void testTranspose();
            void testEqualOperator();
            void testIsEqualOperator();
            void testMultOperator();
            void testSubAssignOperator();

    };
}

#endif // End of MATRIX_TESTING